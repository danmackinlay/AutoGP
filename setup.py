from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os, re
from os.path import join as pjoin
import tensorflow as tf


here_dir = os.path.dirname(os.path.abspath(__file__))


# gcc 4 or using already-built binary,then set USE_CXX11_ABI=1
USE_CXX11_ABI=0
GPU_ARCH="sm_37"


def find_packages(path):
    ret = []
    for root, dirs, files in os.walk(path):
        if '__init__.py' in files:
            ret.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))
    return ret

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

tf_include = tf.sysconfig.get_include()


# find and config CUDA

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path(
            'nvcc',
            os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            return None
            # raise EnvironmentError(
            #    'The nvcc binary could not be '
            #     'located in your $PATH. Either add it to your path,'
            #     'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            return None
            # raise EnvironmentError(
            #     'The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig


# Compile the TensorFlow ops.
compile_command = (
    "g++ -std=c++11 -shared ./autogp/util/tf_ops/vec_to_tri.cc "
    "./autogp/util/tf_ops/tri_to_vec.cc -o ./autogp/util/tf_ops/matpackops.so "
    "-fPIC -I $("
    "python -c 'import tensorflow as tf; "
    "print(tf.sysconfig.get_include())'"
    ")"
)

if sys.platform == "darwin":
    compile_command += " -undefined dynamic_lookup"

os.system(compile_command)



CUDA = locate_cuda()

# ext_modules
bbox = Extension(
    "autogp.utils.bbox",
    ["autogp/utils/bbox.pyx"],
    extra_compile_args={'gcc': ['-Wno-cpp', "-Wno-unused-function"]},
    include_dirs=[numpy_include]
    )
cpu_nms = Extension(
    "autogp.utils.nms.cpu_nms",
    ["autogp/utils/nms/cpu_nms.pyx"],
    extra_compile_args={'gcc': ['-Wno-cpp', "-Wno-unused-function"]},
    include_dirs=[numpy_include]
    )

if CUDA == None:
    roi_pooling = Extension(
        "autogp.networks.layers.roi_pooling_layer.roi_pooling",
        ["autogp/networks/layers/roi_pooling_layer/cpp/roi_pooling_op.cc"],
        extra_compile_args={'gcc': ['-Wno-cpp', "-Wno-unused-function", "-std=c++11", "-shared"]},
        include_dirs=[tf_include]
        )
else:
    gpu_nms = Extension('autogp.utils.nms.gpu_nms',
        ['autogp/utils/nms/nms_kernel.cu', 'autogp/utils/nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_35',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs=[numpy_include, CUDA['include']]
        )

    roi_pooling = Extension(
        "autogp.networks.layers.roi_pooling_layer.roi_pooling",
        [
            "autogp/networks/layers/roi_pooling_layer/cpp/roi_pooling_op_gpu.cu",
            "autogp/networks/layers/roi_pooling_layer/cpp/roi_pooling_op.cc"
        ],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={
            'gcc': [
                '-Wno-cpp',
                "-Wno-unused-function",
                "-std=c++11", "-shared",
                '-D GOOGLE_CUDA=1',
                '-D_GLIBCXX_USE_CXX11_ABI={}'.format(USE_CXX11_ABI)],
            'nvcc': [
                '-arch={}'.format(GPU_ARCH),
                '-std=c++11',
                '-D GOOGLE_CUDA=1',
                '-c',
                '--compiler-options',
                "'-fPIC'",
                '--expt-relaxed-constexpr']},
        include_dirs=[tf_include, CUDA['include']]
    )

if CUDA is None:
    ext_modules = [bbox, cpu_nms, roi_pooling]
else:
    ext_modules = [bbox, cpu_nms, gpu_nms, roi_pooling]


# customize compiler
def customize_compiler(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works."""

    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print extra_postargs

        if CUDA != None and os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        build_ext.build_extensions(self)


setup(
    name='AutoGP',
    version='0.1',
    description='Unified tool for automatric Gaussian PRocess Inference',
    author='Karl Krauth and Edwin Bonilla',
    author_email='edwinbonilla+autogp@gmail.com',
    url='https://github.com/ebonilla/AutoGP',
    license='Apache',
    packages=find_packages('autogp'),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': custom_build_ext
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

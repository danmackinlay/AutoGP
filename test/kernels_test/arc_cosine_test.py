import unittest

import numpy as np
import tensorflow as tf

from autogp import kernels
from ..gaussian_process_test import TestGaussianProcess

SIG_FIGS = 5
RTOL = 10**(-SIG_FIGS)

class TestArcCosine(TestGaussianProcess):

    @classmethod
    def setUpClass(cls):
        super(TestArcCosine, cls).setUpClass()
        cls.session.run(tf.global_variables_initializer())

    def kernel(self, points1, points2=None, degree=0, depth=1):
        arc_cosine = kernels.ArcCosine(degree, depth, white=0.0)
        if points2 is not None:
            return cls.session.run(arc_cosine.kernel(np.array(points1, dtype=np.float32),
                                                      np.array(points2, dtype=np.float32)))
        else:
            return cls.session.run(arc_cosine.kernel(np.array(points1, dtype=np.float32)))

    def diag_kernel(self, points, degree=0, depth=1):
        arc_cosine = kernels.ArcCosine(degree, depth, white=0.0)
        return cls.session.run(arc_cosine.diag_kernel(np.array(points, dtype=np.float32)))

    def test_simple_kern(self):
        kern = self.kernel([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]])
        np.testing.assert_approx_equal(kern, [[1.0, 0.5, 0.5],
                                              [0.5, 1.0, 0.5],
                                              [0.5, 0.5, 1.0]])

    def test_parallel_kern(self):
        kern = self.kernel([[3.0, 5.0, 2.0],
                            [-3.0, -5.0, -2.0],
                            [6.0, 10.0, 4.0]])
        np.testing.assert_approx_equal(kern, [[1.0, 0.0, 1.0],
                                              [0.0, 1.0, 0.0],
                                              [1.0, 0.0, 1.0]])

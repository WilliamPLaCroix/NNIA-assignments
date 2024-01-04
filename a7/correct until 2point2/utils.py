# The eval_numerical_gradient and rel_error functions are taken from HLCV SS2021 Assignment for numerical gradient
# computation.
import numpy as np

"""
DO NOT CHANGE CODE IN THIS FILE
"""

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def correct_scores():
    return np.asarray([[0.10000015, 0.09999991, 0.10000005, 0.09999987, 0.10000014,
                        0.10000028, 0.09999988, 0.1, 0.10000001, 0.0999997],
                       [0.0999998, 0.10000013, 0.09999992, 0.09999989, 0.09999971,
                        0.1000004, 0.09999992, 0.09999986, 0.10000031, 0.10000006],
                       [0.1, 0.1, 0.10000002, 0.10000003, 0.10000006,
                        0.09999995, 0.1000001, 0.09999996, 0.09999985, 0.10000003],
                       [0.10000003, 0.10000014, 0.09999999, 0.10000006, 0.09999987,
                        0.10000021, 0.10000007, 0.09999975, 0.09999988, 0.1],
                       [0.10000002, 0.09999993, 0.10000016, 0.10000005, 0.0999999,
                        0.10000008, 0.1000002, 0.09999988, 0.09999993, 0.09999984]])


def correct_loss():
    return 2.3026622144610953


def init_toy_data(num_inputs=5, input_size=4):
    np.random.seed(23)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y
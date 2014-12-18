# coding=utf-8
__author__ = 'ylei'

from theano import *

import numpy


def test():
    # 只要矩阵中有一个元素是浮点型，其余元素都会用这个类型来存储.
    matrix = numpy.asmatrix([[1., 2], [3, 4], [5, 6]])
    print matrix
    print matrix.shape

    # 访问第２行第０列元素
    print matrix[2, 1]


def create_function():
    import theano.tensor as T

    x = T.dscalar('x')
    y = T.dscalar('y')
    z = x + y
    z.eval({x: 16.3, y: 12.1})


def add_two_matrices():
    import theano.tensor as T

    x = T.dmatrix('x')
    y = T.dmatrix('y')
    z = x * y
    f = function([x, y], z)

    print f([[1, 2], [3, 4]], [[5, 6], [7, 8]])


def exercise_():
    import theano

    a = theano.tensor.vector()

    out = a + a ** 10

    f = theano.function([a], out)

    print f([0, 1, 2])


def exercise_1():
    import theano

    a = theano.tensor.dscalar('a')
    b = theano.tensor.dscalar('b')

    out = a ** 2 + b ** 2 + 2 * a * b

    f = theano.function([a, b], out)

    print f(4, 5)


def exercise():
    import theano

    a = theano.tensor.vector('a')
    b = theano.tensor.vector('b')

    out = a ** 2 + b ** 2 + 2 * a * b

    f = theano.function([a, b], out)

    print f([1, 2], [4, 5])


def create_logistic():
    """
    一次只计算一个输入元素。
    http://deeplearning.net/software/theano/tutorial/examples.html
    这是计算对数函数曲线的y值。输入一个矩阵，元素是x的取值，输出是与输入矩阵中元素对应的y值。
    """
    import theano.tensor as T

    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = function([x], s)
    print logistic([[0, 1], [-1, -2], [-6, 6]])


def calc2elements():
    """
    一次计算两个输入元素。
    http://deeplearning.net/software/theano/tutorial/examples.html
    这是计算对数函数曲线的y值。输入一个矩阵，元素是x的取值，输出是与输入矩阵中元素对应的y值。
    """
    import theano.tensor as T
    from theano import pp
    a, b = T.dmatrices('a', 'b')
    diff = a - b
    abs_diff = abs(diff)
    diff_square = diff ** 2
    f = function([a, b], [diff, abs_diff, diff_square])
    diff, abs_diff, diff_square = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

    print (diff)
    print (abs_diff)
    print (diff_square)

if __name__ == "__main__":
    calc2elements()



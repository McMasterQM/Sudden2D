from problem.SuddenExam import *

import numpy as np
from numpy.testing import assert_equal, assert_allclose


def test_compute_amplitude_1D():
    np.random.seed(42)
    n0 = np.random.randint(1, 100, size=10)
    a = np.random.uniform(size=10)
    m = np.random.randint(1, 100, size=10)
    b = np.random.uniform(size=10)

    answer = []
    for n0_, a_, m_, b_ in zip(n0, a, m, b):
        answer.append(compute_amplitude_1D(int(n0_), int(m_), a_, b_))

    true_answer = [-0.035556844084273884,
                   0.005380850860819553,
                   0.004595920095382628,
                   0.0015983780837765407,
                   -0.010176465319515335,
                   0.0018415025299144165,
                   0.00248657664034064,
                   -0.011588994200529308,
                   -0.005058874434239994,
                   0.0027194446858389947]

    assert_allclose(answer, true_answer)


def test_compute_probability_1D():
    np.random.seed(42)
    n0 = np.random.randint(100, size=10)
    a = np.random.uniform(size=10)
    b = np.random.uniform(size=10)
    x = np.random.uniform(size=10)
    t = np.random.uniform(size=10)

    answer = []
    for n0_, a_, b_, x_, t_ in zip(n0, a, b, x, t):
        answer.append(compute_probability_1D(int(n0_), a_, b_, x_, t_))

    true_answer = [1.2439481729485162,
                   0.0,
                   0.09856732143395193,
                   0.0003413236446248446,
                   0.0,
                   2.198787925093101,
                   0.0,
                   0.0,
                   0.0,
                   0.0]

    assert_allclose(answer, true_answer)


def test_compute_probability_1D():
    np.random.seed(42)
    n0 = np.random.randint(100, size=10)
    a = np.random.uniform(size=10)
    b = np.random.uniform(size=10)
    x = np.random.uniform(size=10)
    t = np.random.uniform(size=10)

    answer = []
    for n0_, a_, b_, x_, t_ in zip(n0, a, b, x, t):
        answer.append(compute_probability_1D(int(n0_), a_, b_, x_, t_))

    true_answer = [1.2439481729485162,
                   0.0,
                   0.09856732143395193,
                   0.0003413236446248446,
                   0.0,
                   2.198787925093101,
                   0.0,
                   0.0,
                   0.0,
                   0.0]

    assert_allclose(answer, true_answer)


def test_compute_probability_1D():
    np.random.seed(42)
    n0 = np.random.randint(100, size=10)
    a = np.random.uniform(size=10)
    b = np.random.uniform(size=10)
    x = np.random.uniform(size=10)
    t = np.random.uniform(size=10)

    answer = []
    for n0_, a_, b_, x_, t_ in zip(n0, a, b, x, t):
        answer.append(compute_probability_1D(int(n0_), a_, b_, x_, t_))

    true_answer = [1.2439481729485162,
                   0.0,
                   0.09856732143395193,
                   0.0003413236446248446,
                   0.0,
                   2.198787925093101,
                   0.0,
                   0.0,
                   0.0,
                   0.0]

    assert_allclose(answer, true_answer)


def test_compute_compute_amplitude_2D():
    np.random.seed(42)
    n0x = np.random.randint(100, size=10)
    n0y = np.random.randint(100, size=10)
    mx = np.random.randint(100, size=10)
    my = np.random.randint(100, size=10)
    ax = 10
    ay = 10
    bx = 15
    by = 15

    answer = []
    for n0x_, n0y_, mx_, my_ in zip(n0x, n0y, mx, my, ):
        answer.append(compute_amplitude_2D(int(n0x_), int(n0y_), int(mx_), int(my_), ax, ay, bx, by))

    true_answer = [-1.971623587769329e-19,
                   2.721146504677133e-19,
                   -0.00022142021407213283,
                   -1.609409172440433e-06,
                   7.340011398478829e-05,
                   -1.1800128390394729e-17,
                   -1.1701638710619274e-19,
                   -9.237924697852752e-19,
                   0.0002630205929916019,
                   6.448508208719624e-18]

    assert_allclose(answer, true_answer)


def test_compute_probability_2D():
    np.random.seed(42)
    n0x = np.random.randint(1, 5, size=10)
    n0y = np.random.randint(1, 5, size=10)
    ax = 1
    ay = 10
    bx = 15
    by = 15
    x = np.random.uniform(1, 5, size=10)
    y = np.random.uniform(1, 5, size=10)
    t = np.random.uniform(0, 1, size=10)

    answer = []
    for n0x_, n0y_, x_, y_, t_ in zip(n0x, n0y, x, y, t):
        answer.append(compute_probability_2D(int(n0x_), int(n0y_), ax, ay, bx, by, x_, y_, t_))

    true_answer = [5.721509633457156e-05,
                   0.0002504849337268151,
                   1.1824076620007244e-07,
                   0.0013808287448141138,
                   0.0005736175132532796,
                   0.0013697593663539393,
                   0.006788567327264091,
                   1.053779388624982e-05,
                   0.003097495119401246,
                   0.06637339244346176]

    assert_allclose(answer, true_answer)


def test_compute_amplitude():
    np.random.seed(42)
    n0 = np.random.randint(1, 100, size=10)
    m = np.random.randint(1, 100, size=10)
    a0 = np.random.uniform(size=10)
    a1 = np.random.uniform(size=10)
    b1 = np.random.uniform(size=10)

    answer = []
    for n0_, m_, a0_, b1_ in zip(n0, m, a0, b1):
        answer.append(compute_amplitude(int(n0_), int(m_), a0_, a0_ + 0.5, b1_))

    true_answer = [0.022513799923826135,
                   -0.0003036740606305687,
                   -0.0002963411336173694,
                   -0.02781900050212731,
                   -0.01729067021973876,
                   -0.022743960632142873,
                   0.0014507921187330403,
                   0.0010207561485302552,
                   0.0026939469114688206,
                   0.0016099661012984215]

    assert_allclose(answer, true_answer)


def test_compute_probability():
    np.random.seed(42)
    n0 = np.random.randint(1, 10, size=10)
    a0 = np.random.uniform(1, 5, size=10)
    b1 = np.random.uniform(4, 10, size=10)
    x = np.random.uniform(1, 5, size=10)
    t = np.random.uniform(0, 1, size=10)

    answer = []
    for n0_, a0_, b1_, x_, t_ in zip(n0, a0, b1, x, t):
        answer.append(compute_probability(int(n0_), a0_, a0_ + 0.5, b1_, x_, t_))

    true_answer = [0.3979718816316191,
                   0.1857582904035265,
                   0.04968086175230966,
                   0.6732527586462362,
                   0.010013001828320782,
                   0.1842225789301185,
                   0.3228675950718869,
                   0.13072785373064183,
                   0.035173228468738355,
                   0.006520284001852621]

    assert_allclose(answer, true_answer)

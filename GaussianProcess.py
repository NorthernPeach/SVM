import random
import math
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def read_matrix(filename='input.data'):
    data = open(filename, 'r').readlines()
    data = np.array([row.strip().split() for row in data], dtype='float')
    return data

def rtnl_qdr_kernel(x1, x2, a, l, var = 5):
    square = cdist(x1, x2, metric='sqeuclidean')
    kernel = (1 + square / (2* a * l**2)) ** (- a)
    return kernel * var

def draw(m, U, B, Optim):
    if Optim:
        plt.title('Gausian Process with Optimization')
    else:
        plt.title('Gausiian Process')
    plt.xlim(-60, 60)
    plt.scatter(X, Y, c='r', label='Training data')
    plt.plot(X_test.ravel(), m.ravel(), 'b', label='Mean of f')
    plt.fill_between(X_test.ravel(), U, B, alpha=0.3, label='Confidence interval')
    plt.legend(fontsize='x-small')
    plt.show()


def GP(a, l, var = 5, Optim=False):
    C = rtnl_qdr_kernel(X, X, a, l, var) 
    K_star = rtnl_qdr_kernel(X_test, X_test, a, l, var) + np.eye(n) * 1/b
    K_x_star = rtnl_qdr_kernel(X, X_test, a, l, var)

    m = (K_x_star.T).dot(np.linalg.inv(C)).dot(Y)
    var = K_star - (K_x_star.T).dot(np.linalg.inv(C)).dot(K_x_star)

    # Set bounds for the confidence interval
    up_bound = np.zeros(n)
    bot_bound = np.zeros(n)
    for i in range(n):
        up_bound[i] = m[i, 0] + var[i, i] * 1.96
        bot_bound[i] = m[i, 0] - var[i, i] * 1.96

    # Draw mean and the interval
    draw(m, up_bound, bot_bound, Optim)


def Optimize():
    def marginal_LH(guess, X, Y):
        C = rtnl_qdr_kernel(X, X, guess[0], guess[1], guess[2])
        return (0.5 * Y.T.dot(np.linalg.inv(C)).dot(Y)
                + 0.5 * np.log(np.linalg.det(C))
                + n_items / 2 * math.log(2.0 * math.pi))[0]

    guess = [1.0, 1.0, 1.0]
    result = minimize(fun=marginal_LH, x0=guess, args=(X, Y))
    a_opt, l_opt, var_opt = result.x
    GP(a_opt, l_opt, var_opt, True)

b = 5
n = 1000
a = 1
l = 1
dataset = read_matrix()
X, Y = dataset[:, 0].reshape(-1, 1), dataset[:, 1].reshape(-1, 1)
X_test = np.linspace(-60, 60, n).reshape(-1, 1)
n_items = X.shape[0]

GP(a, l)
Optimize()
import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import *

def get_data():
    X = np.genfromtxt('X_train.csv', delimiter=',')
    Y = np.genfromtxt('Y_train.csv', delimiter=',')

    X_test = np.genfromtxt('X_test.csv', delimiter=',')
    Y_test = np.genfromtxt('Y_test.csv', delimiter=',')
    return X, Y, X_test, Y_test

def svm(kernel_type, command):
    print("\n", kernel_type)
    command = command + " -q"  # -q: quiet mode (no outputs); -s svm_type: default=0 -- C_SVC (multi-class classification)
    param = svm_parameter(command)
    prob = svm_problem(Y_train, X_train)
    model = svm_train(prob, param)
    _, test_acc, _ = svm_predict(Y_test, X_test, model)  # p_label, p_acc, p_val
    return test_acc[0]

def gridsearch_svm(kernel_type, command, fold=0):
    print("\n", kernel_type)
    if fold:
        command = command + " -v {}".format(fold)   # -v n: n-fold cross validation mode
    command = command + " -q"  # -s svm_type: default=0 -- C_SVC
    param = svm_parameter(command)
    prob = svm_problem(Y_train, X_train)
    val_acc = svm_train(prob, param)
    return val_acc

def GridSearch(kernel_type):
    cost = [1e-2, 1, 10, 1e2]
    degree = [1, 2, 3, 4, 5]
    gamma = [1, 1e-1, 1e-2]
    coef0 = [1, 4, 5]
    n = 3

    acc_record = []
    max_acc = 0.0
    if kernel_type == "Linear":
        for ci in range(len(cost)):
            command = " -c " + str(cost[ci])
            acc = gridsearch_svm(
                kernel_type+command, "-t 0"+command, fold=n)
            if acc > max_acc:
                max_acc = acc
                max_param = command

    elif kernel_type == "Polynomial":
        for ci in range(len(cost)):
            command = " -c "+str(cost[ci])
            for ri in range(len(coef0)):
                command_r = command + " -r " + str(coef0[ri])
                for gi in range(len(gamma)):
                    command_g = command_r + " -g " + str(gamma[gi])
                    for di in range(len(degree)):
                        command_d = command_g + " -d " + str(degree[di])
                        acc = gridsearch_svm(
                            kernel_type+command_d, "-t 1"+command_d, fold=n)
                        if acc > max_acc:
                            max_acc = acc
                            max_param = command_d

    elif kernel_type == "RBF":
        for ci in range(len(cost)):
            command = " -c " + str(cost[ci])
            for gi in range(len(gamma)):
                command_g = command + " -g " + str(gamma[gi])
                acc = gridsearch_svm(
                    kernel_type+command_g, "-t 2"+command_g, fold=n)
                if acc > max_acc:
                    max_acc = acc
                    max_param = command_g
                    
    print(".............................................")
    print(kernel_type+" model performance")
    print("Best Parameters:", max_param)
    print("Max accuracy", max_acc)
    svm("Testing "+kernel_type, max_param)
    print(".............................................")

def linear_kernel(x1, x2):
    return x1.dot(x2.T)

def RBF_kernel(x1, x2, gamma):
    return np.exp(gamma * cdist(x1, x2, metric='sqeuclidean'))

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_data()
    parts = [1, 2, 3]
    for part in parts:
        # ............ part I ............
        if part == 1:
            svm("Linear", "-t 0")  # -c cost
            svm("Polynomial", "-t 1")  # -c, -r coef0, -g gamma, -d degree
            svm("RBF", "-t 2")  # -c, -g

        # ............ part II ............
        if part == 2:
            GridSearch("Linear")
            GridSearch("Polynomial")
            GridSearch("RBF")

        # ............ part III ............
        if part == 3:
            cost = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
            gamma = [1 / 784, 1, 1e-1, 1e-2, 1e-3, 1e-4]
            x = X_train
            row, col = x.shape
            max_acc = 0.0
            gamma_best = 0
            linear_k = linear_kernel(x, x)
            for gi in range(len(gamma)):
                rbf_k = RBF_kernel(x, x, -gamma[gi])
                my_k = linear_k + rbf_k
                my_k = np.hstack((np.arange(1, row + 1).reshape(-1, 1), my_k))
                prob = svm_problem(Y_train, my_k, isKernel=True)
                for ci in range(len(cost)):
                    command = "-t 4 -c " + str(cost[ci]) + " -v 3 -q"
                    param_rec = "-t 4 -c " + str(cost[ci]) + " -q"
                    print("-g", gamma[gi], command)
                    param = svm_parameter(command)
                    val_acc = svm_train(prob, param)

                    if val_acc > max_acc:
                        max_acc = val_acc
                        max_param = param_rec
                        gamma_best = gi
            print(".............................................")
            print("Best Parameters:", " -g", gamma[gamma_best], max_param)
            print("Max accuracy:", max_acc)

            rbf_k = RBF_kernel(x, x, -gamma[gamma_best])
            my_k = linear_k + rbf_k
            my_k = np.hstack((np.arange(1, row + 1).reshape(-1, 1), my_k))
            prob = svm_problem(Y_train, my_k, isKernel=True)
            param = svm_parameter(max_param)
            model = svm_train(prob, param)

            row, col = X_test.shape
            linear_k = linear_kernel(X_test, X_test)
            rbf_k = RBF_kernel(X_test, X_test, -gamma[gamma_best])
            my_k = linear_k + rbf_k
            my_k = np.hstack((np.arange(1, row + 1).reshape(-1, 1), my_k))
            _, test_acc, _ = svm_predict(Y_test, my_k, model)
            print(".............................................")

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):  # learninRate正则化参数（lambda）
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    cross_cost = np.multiply(-y, np.log(sigmoid(X * theta.T))) - np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # 正则化部分
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    whole_cost = np.sum(cross_cost) / len(X) + reg
    return whole_cost


def gradientDescent(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    # 计算预测误差
    error = sigmoid(X * theta.T) - y
    # 计算梯度
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

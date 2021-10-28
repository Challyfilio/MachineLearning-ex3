import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):  # learninRate正则化参数（lambda）
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    cross_cost = np.multiply(-y, np.log(sigmoid(X * theta.T))) \
                 - np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # 正则化部分
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[1:], 2))
    whole_cost = np.sum(cross_cost) / len(X) + reg
    return whole_cost


def gradientDescent(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])  # theta的数量
    # 计算预测误差
    error = sigmoid(X * theta.T) - y
    # 计算梯度
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    return np.array(grad).ravel()

#一对多分类器
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params + 1))

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradientDescent)
        all_theta[i - 1, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # 计算样本属于每一类的概率
    h = sigmoid(X * all_theta.T)
    print(h)
    # 每个样本中预测概率最大值
    h_argmax = np.argmax(h, axis=1)
    # 索引+1
    h_argmax = h_argmax + 1
    return h_argmax


if __name__ == '__main__':
    #数据集
    data = loadmat("ex3data1.mat")
    print(data)

    print(data['X'].shape, data['y'].shape)

    # 数据可视化
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_images = data['X'][sample_idx, :]
    print(sample_images)

    fig, ax = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

    rows = data['X'].shape[0]
    params = data['X'].shape[1]
    all_theta = np.zeros((10, params + 1))
    X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

    theta = np.zeros(params + 1)

    y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
    y_0 = np.reshape(y_0, (rows, 1))

    print(X.shape, y_0.shape, theta.shape, all_theta.shape)
    # 标签
    print(np.unique(data['y']))

    all_theta = one_vs_all(data['X'], data['y'], 10, 1)
    # print(all_theta)

    y_pred = predict_all(data['X'], all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct))) / float(len(correct))
    print('accuracy=' + str(accuracy))
    print(classification_report(data['y'],y_pred))

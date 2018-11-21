import numpy as np
import math
import json
import random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
LEARNING_RATE = 0.0005

ACTIVATION_RELU = 1
ACTIVATION_SIGMOID = 2
ACTIVATION_SOFTMAX = 3

LOSSFUNCTION_MSE = 1
LOSSFUNCTION_CROSS_ENTROPY = 2

class Layer:
    deltas = np.array(0)
    activation = 0
    nodes = 0
    net = np.array(0)
    o = net
    def __init__(self, _nodes):
        self.nodes = _nodes
        self.net = np.zeros((self.nodes))

class InputLayer(Layer):
    def __init__(self, _nodes):
        self.nodes = _nodes
        self.net = np.zeros((self.nodes))
    def setO(self, _input):
        self.o = _input
    def forward(self, _):
        #
        #self.net = np.dot(self.w.transpose(), _inputLayer.o) + self.b
        None
    def activate(self, _):
        None
    def backward(self, _):
        #
        None


class NeuralLayer(Layer):
    o = np.array(0)
    w = np.array(0)
    b = np.array(0)
    
    def __init__(self, _nodeCount, _inputCount):
        #
        self.nodes = _nodeCount
        self.net = np.zeros((self.nodes, 1), np.float)
        self.o = np.zeros((self.nodes, 1), np.float)
        self.deltas = np.zeros((self.nodes, _inputCount), np.float)
        self.w = np.random.normal(0, 1, (self.nodes, _inputCount))
        self.b = np.random.normal(0, 1, (self.nodes))
    
    def setActivation(self, _act = None):
        self.activation = _act
    def forward(self, _inputLayer):
        # 
        self.net = np.matmul(self.w, _inputLayer.o) + self.b
    def activate(self, _activationFunction):
        self.o = np.fromiter((_activationFunction(xi) for _, xi in enumerate(self.net.tolist())), self.net.dtype)
    def backward(self, _inputLayer, _outputLayer, _deriveFunction):
        # this.delta = out.w * out.delta * f'(this.net)
        # this.w += LEARNINGRATE * this.input * this.delta
        global LEARNING_RATE
        tmp = self.net.tolist()
        f_dot = np.fromiter((_deriveFunction(xi) for _, xi in enumerate(self.net.tolist())), self.net.dtype)
        self.deltas = np.matmul(_outputLayer.w.transpose(), _outputLayer.deltas) * f_dot
        #self.deltas = np .reshape(self.deltas, (-1, 1))
        self.w += LEARNING_RATE * _inputLayer.o * self.deltas.reshape(-1, 1)
    def setDelta(self, _delta):
        self.deltas = _delta

class OutputLayer(NeuralLayer):
    def backward(self, _inputLayer, _, _deriveFunction):
        global LEARNING_RATE
        self.w += LEARNING_RATE * _inputLayer.o * self.deltas.reshape(-1, 1)
        

class Trainer:
    dataDir = ''
    testSet = [] # 0: input, 1: output
    trainSet = []
    input_layer = []
    output_layer = []
    layers = [] # It has input, output layer too.
    def __init__(self, _networkTopology, _activation = ACTIVATION_RELU):
        self.input_layer = InputLayer(_networkTopology[0])
        self.layers.append(self.input_layer)
        for idx, n in enumerate(_networkTopology[1:-1:]):
                self.layers.append(NeuralLayer(n, _networkTopology[idx]))
                self.layers[-1].setActivation(_activation)
        self.output_layer = OutputLayer(_networkTopology[-1], _networkTopology[-2])
        self.output_layer.setActivation(_activation)
        self.layers.append(self.output_layer)
    def setTrainSet(self, _data):
        tmp = json.loads(_data)
        self.trainSet.append(np.array(tmp['input']))
        self.trainSet.append(np.array(tmp['output']))
    def saveParam(self):
        dataDir = ''
    # Activation functions
    def ReLU(self, x):
        if x < 0:
            return 0
        else:
            return x
    def ReLU_derive(self, x):
        if x < 0:
            return 0
        else:
            return 1
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def sigmoid_derive(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    # Loss functions
    def MSE(self, x, y):
        return -1 * ((x - y) * (x - y)) / 2
    def MSE_derive(self, x, y):
        return y - x
    def randDraw(self, _dataset, _stochastic):
        ret = []
        ret.append([]) # for input
        ret.append([]) # for output
        for idx, r in enumerate(_dataset[0]):
            if (rnd.random() < _stochastic):
                ret[0].append(r)
                ret[1].append(_dataset[1][idx])
        return ret
    def forward(self, _row):
        self.input_layer.setO(_row)
        for idx, layer in enumerate(self.layers):
            layer.forward(self.layers[idx - 1])
            if layer.activation == ACTIVATION_RELU:
                layer.activate(self.ReLU)
            elif layer.activation == ACTIVATION_SIGMOID:
                layer.activate(self.sigmoid)
            # elif layer.activation == ACTIVATION_SOFTMAX:
            #     layer.activate(self.softmax)
    def calcLoss(self, _lossFunction, _y):
        delta = 0
        if _lossFunction == LOSSFUNCTION_MSE:
            tmp = self.output_layer.o.tolist()
            delta = np.fromiter((self.MSE_derive(xi, _y[idx]) for idx, xi in enumerate(tmp)), self.output_layer.o.dtype)
        self.output_layer.setDelta(delta)
        return delta
    def updateWeight(self):
        #     def backward(self, _inputLayer, _outputLayer, _deriveFunction):
        tmp = self.layers[::-1]
        for idx, layer in enumerate(tmp[:-1:]):
            if layer.activation == ACTIVATION_RELU:
                layer.backward(tmp[idx + 1], tmp[idx - 1], self.ReLU_derive)
            elif layer.activation == ACTIVATION_SIGMOID:
                layer.backward(tmp[idx + 1], tmp[idx - 1], self.sigmoid_derive)
    def predict(self):
        s = 0
        for idx, r in enumerate(self.trainSet[0]):
            self.forward(r)
            _y = self.trainSet[1][idx]
            tmp = self.output_layer.o.tolist()
            delta = np.fromiter((self.MSE(xi, _y[idx]) for idx, xi in enumerate(tmp)), self.output_layer.o.dtype)
            s += delta.sum()
        return s / self.trainSet[0].__len__()
    def SGD(self, _iteration, _lossFunction = LOSSFUNCTION_MSE, _stochastic = 0.1):
        for i in range(_iteration):
            tmpSet = self.randDraw(self.trainSet, _stochastic)
            num = tmpSet[0].__len__() 
            if num > 0:
                delta_sum = 0
                for idx, r in enumerate(tmpSet[0]):
                    self.forward(r)
                    delta_sum += self.calcLoss(_lossFunction, tmpSet[1][idx])
                self.layers[-1].setDelta(delta_sum / num)
                self.updateWeight()
    def GD(self, _iteration, _lossFunction = LOSSFUNCTION_MSE):
        for i in range(_iteration):
            for idx, r in enumerate(self.trainSet[0]):
                self.forward(r)
                self.calcLoss(_lossFunction, self.trainSet[1][idx])
                self.updateWeight()
    def predWithData(self, _X):
        #_X is matrix. column: x1, x2, x3 ... .
        ret = _X.copy()
        num = _X.shape[0]
        o = []
        for idx,r in enumerate(_X):
            self.forward(r)
            o.append(self.output_layer.o)  
        return np.concatenate((ret, o), axis = 1)


def test(_iter, _path, _topology, _title, _loss = LOSSFUNCTION_MSE, _act = ACTIVATION_RELU):
    net = Trainer(_topology)
    f = open(_path, 'r')
    data = f.read()
    net.setTrainSet(data)
    pred = []
    delta = []
    pred.append(-1*net.predict())
    iter = _iter
    it = 0
    for i in range(iter):
        #net.SGD(10,_stochastic= 0.1)
        net.GD(1)
        pred.append(-1*net.predict())
        tmp = 0
        delta.append([])
        for _, l in enumerate(net.layers):
            delta[-1].append(l.deltas.sum().mean())
        if i % 50 == 0:
            print("ITER: " + str(int((i/iter) * 100)) + "% | --- | pred: " + str(pred[-1]))
            if (str(pred[-1]) == 'nan'):
                it = 100000
                break
            if (pred[-1] < 0.01):
                it = i
                break
        
    fig = plt.figure(figsize=(10,5))
    plt.subplot(121)
    fig.add_subplot(121).plot(pred, 'ro')
    
    # Plot output about all X.
    scale = 40
    x1 = np.linspace(-0.5, 1.5, scale).reshape(-1,1)
    x2 = np.linspace(-0.5, 1.5, scale).reshape(-1,1)
    X = np.zeros((scale*scale, 2))
    for i, r1 in enumerate(x1):
        for j, r2 in enumerate(x2):
            X[i*scale + j][0] = r1
            X[i*scale + j][1] = r2

    pred2 = net.predWithData(X)

    ax = fig.add_subplot(122, projection = '3d')
    ax.scatter(pred2.T[0], pred2.T[1], pred2.T[2])

    plt.title(_title)
    #plt.show()
    #plt.savefig('./fig/1')
    delta = np.array(delta).T
    # for idx, ld in enumerate(delta):
    #     delta[idx] = np.mean(ld)
    
    
    plt.ylim(-0.5, 0.5)
    tmp = []
    for i in range(_topology.__len__()):
        plt.subplot(_topology.__len__(),1,i+1)
        plt.title('AVG(delta) of ' + str(i+1) + ' Layer')
        plt.plot(delta[i])
        tmp.append(delta[i].mean())
    print(tmp)
    #plt.show()
    #plt.savefig('./fig/2')
    return it





netTopology = [2,8,8,1]
top = []
top.append([2,2,1])
top.append([2,4,1])
top.append([2,8,1])
top.append([2,2,2,1])
top.append([2,4,4,1])
top.append([2,8,4,1])
p = './data/'
path_xor = p + '/XOR/trainset.json'
path_and = p + '/AND/trainset.json'
path_or = p + '/OR/trainset.json'
path_doughnut = p + '/doughnut/trainset.json'
path_doughnut2 = p + '/doughnut/trainset2.json'
paths = []
paths.append(path_and)
paths.append(path_or)
paths.append(path_xor)
paths.append(path_doughnut)
t = []
for i in range(10):
    t.append(test(20000, path_doughnut, [2,8,8,8,1], ''))#, _act = ACTIVATION_SIGMOID)
print(t / 10)
# for i, p in enumerate(paths):
#     for j, t in enumerate(top):
#         title = str(i) + '-' + str(j) + ': ' + str(t)
#         test(p, t, title)


# # Trainer initiation: (self, _networkTopology, _activation = ACTIVATION_RELU)
# net = Trainer(netTopology)
# # path = 'C:/Users/hjan1/Documents/JJA/projects/MLP_py/data/doughnut/trainset.json'
# path = './data/doughnut/trainset.json'

# f = open(path, 'r')
# data = f.read()
# net.setTrainSet(data)
# pred = []
# pred.append(net.predict())

# iter = 500
# for i in range(iter):
#     # net.SGD(1,_stochastic= 0.1)
#     net.GD(10)
#     if i % 50 == 0:
#         print("ITER: " + str(int((i/iter) * 100)) + "% | --- | pred: " + str(-1*pred[-1]))
#     pred.append(net.predict())


# fig = plt.figure(figsize=(10,5))
# plt.subplot(121)
# fig.add_subplot(121).plot(pred, 'ro')

# # Plot output about all X.
# scale = 40
# x1 = np.linspace(-1, 2, scale).reshape(-1,1)
# x2 = np.linspace(-1, 2, scale).reshape(-1,1)
# X = np.zeros((scale*scale, 2))
# for i, r1 in enumerate(x1):
#     for j, r2 in enumerate(x2):
#         X[i*scale + j][0] = r1
#         X[i*scale + j][1] = r2

# pred2 = net.predWithData(X)

# ax = fig.add_subplot(122, projection = '3d')
# ax.scatter(pred2.T[0], pred2.T[1], pred2.T[2])

# plt.show()
import numpy as np
import math
import json
import random as rnd
import matplotlib.pyplot as plt
LEARNING_RATE = 0.0005

ACTIVATION_RELU = 1
ACTIVATION_SIGMOID = 2
ACTIVATION_SOFTMAX = 3

LOSSFUNCTION_MSE = 1
LOSSFUNCTION_CROSS_ENTROPY = 2

class Layer:
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

# class ActivationLayer(Layer):
#     deltas = np.array(0)

#     def __init__(self, _nodeCount):
#         deltas = np.zeros((_nodeCount))

# class ReLULayer(ActivationLayer):

#     def activation(self, _input):
#         ret = 0 if _input < 0 else ret = _input
#         return ret

#     def derive(self, _input):
#         ret = 0 if _input < 0 else ret = 1
#         return ret
    
#     def forward(self, _inputLayer):
#         self.net = np.fromiter((activation(xi) for xi in _inputLayer.net), x.dtype)

#     def backward(self, _outputLayer):


class NeuralLayer(Layer):
    deltas = np.array(0)
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
        self.w -= LEARNING_RATE * _inputLayer.o * self.deltas.reshape(-1, 1)
        

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
    def setTrainSet(self, _path, ):
        f = open(_path, 'r')
        data = f.read()
        tmp = json.loads(data)
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
        return (x - y) * (x - y) / 2
    def MSE_derive(self, x, y):
        return x - y
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
            _y = self.trainSet[1]
            tmp = self.output_layer.o.tolist()
            delta = np.fromiter((self.MSE(xi, _y[idx]) for idx, xi in enumerate(tmp)), self.output_layer.o.dtype)
            s += delta.sum()
        return s / self.trainSet[0].__len__()
    def SGD(self, _iteration, _lossFunction = LOSSFUNCTION_MSE, _stochastic = 0.1):
        for i in range(_iteration):
            tmpSet = self.randDraw(self.trainSet, _stochastic)
            num = tmpSet[0].__len__() 
            delta_sum = 0
            for idx, r in enumerate(tmpSet[0]):
                self.forward(r)
                delta_sum += self.calcLoss(_lossFunction, tmpSet[1][idx])
            self.layers[-1].setDelta(delta_sum / num)
            self.updateWeight()





netTopology = [2,2,3,1]

#(self, _networkTopology, _activation = ACTIVATION_RELU)
net = Trainer(netTopology)
net.setTrainSet('C:/Users/hjan1/Documents/JJA/projects/MLP_py/data/doughnut/trainset.json')
#net.setTrainSet('./data/doughnut/trainset.json')
pred = []
pred.append(net.predict())

for i in range(1000):
    net.SGD(10,_stochastic= 0.4)
    pred.append(net.predict())

plt.plot(, pred, 'ro')
plt.show()
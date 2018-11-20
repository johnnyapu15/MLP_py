import numpy as np
LEARNING_RATE = 0.0005

ACTIVATION_RELU = 1
ACTIVATION_SIGMOID = 2
ACTIVATION_SOFTMAX = 3

class Layer:
    nodes = 0
    net = np.array(0)
    def __init__(self):
        self.nodes = 0

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
    activation = 0
    deltas = np.array(0)
    o = np.array(0)
    w = np.array(0)
    b = np.array(0)
    
    def __init__(self, _nodeCount, _inputLayer):
        #
        self.nodes = _nodeCount
        self.net = np.zeros((self.nodes), np.float)
        self.o = np.zeros((self.nodes), np.float)
        self.deltas = np.zeros((_inputLayer.nodes, self.nodes), np.float)
        self.w = np.random.normal(0, 1, (_inputLayer.nodes, self.nodes))
        self.b = np.random.normal(0, 1, (self.nodes))
        
    def setActivation(self, _act):
        self.activation = _act
    def forward(self, _inputLayer):
        #
        self.net = np.dot(self.w.transpose(), _inputLayer.o) + self.b
    def activate(self, _activationFunction):
        self.o = np.fromiter((_activationFunction(xi) for xi in self.net), self.net.dtype)
    def backward(self, _outputLayer):
        #
        global LEARNING_RATE

class Trainer:
    dataDir = ''
    testSet = np.array(0)
    trainSet = np.array(0)
    
    def setData(self, _):
        _
    
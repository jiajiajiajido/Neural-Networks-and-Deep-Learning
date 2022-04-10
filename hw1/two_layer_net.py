# -*- coding:utf-8 -*-

import numpy as np
from Layers import *
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_decay_lambda=0):
        # He初始值初始化权重
        self.params = {}
        self.params['W1'] = np.sqrt(2.0 / input_size) * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.sqrt(2.0 / hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.weight_decay_lambda = weight_decay_lambda

        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # loss引入了L2正则化
    def loss(self, x, t):
        y = self.predict(x)
        penalty = 0
        for idx in range(1, 3):
            W = self.params['W' + str(idx)]
            m = y.shape[0]
            penalty += (1 / m) * 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.lastLayer.forward(y, t) + penalty

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
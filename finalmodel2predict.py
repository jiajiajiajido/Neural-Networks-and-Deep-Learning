# -*- coding:utf-8 -*-

import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
import pickle

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

with open('save/lr0.5weight0.01hidden200.pickle', 'rb') as f:
    network = pickle.load(f)

print("acc=", network.accuracy(x_test, t_test))

print(network.params)


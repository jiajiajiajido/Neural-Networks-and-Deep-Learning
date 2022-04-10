# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from two_layer_net import TwoLayerNet
from Functions import shuffle_dataset
from trainer import Trainer
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# x_train, t_train = x_train[:500], t_train[:500]
# 分割验证集
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, hidden, epocs=20):
    network = TwoLayerNet(input_size=784, hidden_size=hidden, output_size=10, weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs, mini_batch_size=100,
                      lr=lr, decay=0.8, verbose=False)
    trainer.train()
    return trainer.test_loss_list, trainer.train_loss_list, trainer.test_acc_list, trainer.train_acc_list, trainer.network


# 超参数的随机搜索======================================
results_val = {}
results_train = {}
loss_val = {}
loss_train = {}

lr_list = [0.01, 0.1, 0.5]
weight_decay_list = [0.001, 0.01, 0.1]
hidden_list = [50, 100, 200]


optimization_trial = len(lr_list) * len(weight_decay_list)
for i in range(len(lr_list)):
    for j in range(len(weight_decay_list)):
        for k in range(len(hidden_list)):
            lr = lr_list[i]
            weight_decay = weight_decay_list[j]
            hidden = hidden_list[k]

            val_loss_list, train_loss_list, val_acc_list, train_acc_list, net = __train(lr, weight_decay, hidden)
            with open(f'save/lr{lr}weight{weight_decay}hidden{hidden}.pickle', 'wb') as f:
                pickle.dump(net, f)
            print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay) + ", hidden:" + str(hidden))
            key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay) + "hidden:" + str(hidden)
            results_val[key] = val_acc_list
            results_train[key] = train_acc_list

            print("val loss:" + str(val_loss_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay) + ", hidden:" + str(hidden))
            loss_val[key] = val_loss_list
            loss_train[key] = train_loss_list

# 绘制acc图========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

plt.figure(figsize=(20, 15))
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    plt.legend(['test_acc', 'train_acc'])
    i += 1

    if i >= graph_draw_num:
        break

plt.show()


# 绘制对应loss图========================================================
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
i = 0

plt.figure(figsize=(20, 15))
for key, val_acc_list in sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True):
    # print("Best-" + str(i+1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)
    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-" + str(i+1))
    plt.ylim(0.0, 1.0)
    if i % 5: plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_loss_list))
    plt.plot(x, val_loss_list)
    plt.plot(x, loss_train[key], "--")
    plt.legend(['test_loss', 'train_loss'])
    i += 1

    if i >= graph_draw_num:
        break

plt.show()


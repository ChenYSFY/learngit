#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by yyjn on 16-11-13

import tensorflow as tf
import numpy as np
from   tensorflow.examples.tutorials.mnist import input_data

# create data
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 0.1 + 0.13
#
# ### create tensorflow struvture start
# Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# biases = tf.Variable(tf.zeros([1]))
# y = Weights * x_data + biases
# loss = tf.reduce_mean(tf.square(y - y_data))
# # train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# # train = tf.train.GradientDescentOptimizer(0.5)
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# sess.run(train)
#
# for step in range(101):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(Weights), sess.run(biases))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
if __name__ == '__main__':
    pass

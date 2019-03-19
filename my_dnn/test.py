import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from my_dnn_models import MyNN
import time
rng = np.random

learning_rate = 0.01
training_epochs = 2000
display_step = 50

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]


def linear_regression():
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(rng.randn(), name='Weight')
    b = tf.Variable(rng.randn(), name='Bias')

    activation = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: {0} cost={1} W={2} b={3}".format(epoch + 1,
                                                               sess.run(cost, feed_dict={X: train_X, Y:train_Y}),
                                                               sess.run(W),
                                                               sess.run(b)))

        print("Optimization Finished!")
        print("cost={0} W={1} b={2}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y}),
                                            sess.run(W),
                                            sess.run(b)))

        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()


def gradient():
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = tf.constant([[1], [2]], dtype=tf.float32)

    w = tf.get_variable('weight', shape=[2, 1], initializer=tf.constant_initializer([[1], [2]], dtype=tf.float32))
    b = tf.get_variable('bias', shape=[1], initializer=tf.constant_initializer([1], dtype=tf.float32))

    with tf.GradientTape() as tape:
        cost = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x, w) + b - y))

    w_grad, b_grad = tape.gradient(cost, [w, b])
    print([cost.numpy(), w_grad.numpy(), b_grad.numpy()])
    s = tf.square(tf.matmul(x, w) + b - y)
    #
    tmp = tf.matmul(tf.transpose(x), (tf.matmul(x, w) + b - y))
    print(tmp)
    ss = tf.reduce_sum(s)
    print(ss.numpy())


def linear_regression2():
    epochs = 10000

    X_raw = np.array([2013, 2014, 2015, 2016, 2017])
    y_raw = np.array([12000, 14000, 15000, 16500, 17500])
    X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    w = tf.get_variable('weight', dtype=tf.float32, shape=[1], initializer=tf.random_normal_initializer)
    b = tf.get_variable('bias', dtype=tf.float32, shape=[1], initializer=tf.random_normal_initializer)
    variables = [w, b]
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)

    for i in range(epochs):
        with tf.GradientTape() as tape:
            y_pred = w * X + b
            cost = tf.reduce_sum(tf.square(y_pred - y))

        grads = tape.gradient(cost, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

        if (i + 1) % 500 == 0:
            print("Epochs: {0}, cost={1}, w={2}, b={3}".format(i + 1, cost.numpy(), w.numpy()[0], b.numpy()[0]))

    plt.plot(X, y, 'ro', label='Original data')
    plt.plot(X, w * X + b, label='Fitted line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    '''x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    y = [[10.0], [20.0]]

    lr = LinearRegression()
    lr.input_data(x, y)
    lr()'''

    m = MyNN()
    m.train()
    # m.restore_model()
    # m.model_test()


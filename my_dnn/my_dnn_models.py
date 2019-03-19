import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
tf.enable_eager_execution()


class FullConnectionLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=1, kernel_initializer=tf.zeros_initializer(),
                                           bias_initializer=tf.zeros_initializer())

    def __call__(self, training_data):
        output = self.dense(training_data)

        return output


class LinearRegression:
    __learning_rate = 1e-3
    __epochs = 10000
    __steps = 500

    w = tf.get_variable('weights', shape=[1], dtype=tf.float32, initializer=tf.random_normal_initializer)
    b = tf.get_variable('bias', shape=[1], dtype=tf.float32, initializer=tf.random_normal_initializer)
    variables = [w, b]

    model = FullConnectionLayer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=__learning_rate)

    __x_train = tf.constant([0])
    __y_train = tf.constant([0])

    def __init__(self, rate=1e-3, epochs=10000, steps=500):
        self.__learning_rate = rate
        self.__epochs = epochs
        self.__steps = steps

    def __repr__(self):
        string1 = 'weights:\n{0}'.format(self.model.variables[0].numpy())
        string2 = 'bias: {0}'.format(self.model.variables[1].numpy())

        return string1 + '\n' + string2
    __str__ = __repr__

    def __call__(self):
        self.train()
        self.show_result()
        print(self)

    def __pre_process(self):
        # self.__x_train = (self.__X - self.__X.min()) / (self.__X.max() - self.__X.min())
        # self.__y_train = (self.__Y - self.__Y.min()) / (self.__Y.max() - self.__Y.min())
        # self.__x_train = tf.constant(self.__x_train)
        # self.__y_train = tf.constant(self.__y_train)
        pass

    def input_data(self, X, Y):
        print(X, Y)
        if isinstance(X, list) and len(X) != 0:
            self.__x_train = tf.constant(X)
        if isinstance(Y, list) and len(Y) != 0:
            self.__y_train = tf.constant(Y)

    def show_result(self):
        '''plt.plot(self.__x_train, self.__y_train, 'ro', label='Original data')
        plt.plot(self.__x_train, self.w * self.__x_train + self.b, label='Fitted line')
        plt.legend()
        plt.show()'''
        pass

    def train(self):
        self.__pre_process()

        for i in range(self.__epochs):
            with tf.GradientTape() as tape:
                y_pred = self.model(self.__x_train)
                cost = 0.5 * tf.reduce_sum(tf.square(y_pred - self.__y_train))

            grads = tape.gradient(cost, self.model.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))

            if (i + 1) % self.__steps == 0:
                print("""Epochs: {0}, cost={1}""".format(i + 1, cost.numpy()))

        print('Training finished')


"""
/**********************************************************************************************************************/
/*********************************************** Implementation of DNN ************************************************/
/**********************************************************************************************************************/
"""


class DataLoader:
    def __init__(self):
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        self.train_data = mnist.train.images # np.array [55000,!784]
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32) # np.array [55000] of,!int32
        self.eval_data = mnist.test.images # np.array [10000,784]
        self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32) # np.array [10000] of!int32

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)

        return self.train_data[index, :], self.train_labels[index]


class MLP(tf.keras.Model):
    def __init__(self, first_layer=100, second_layer=10):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(units=first_layer, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=second_layer)

    def __call__(self, inputs):
        layer1_output = self.layer1(inputs)
        layer2_output = self.layer2(layer1_output)

        return layer2_output

    def predict(self, inputs):
        outputs = self(inputs)

        return tf.argmax(outputs, axis=-1)


class MyCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.__conv1 = tf.keras.layers.Conv2D(filters=32,
                                              kernel_size=[5, 5],
                                              padding="same",
                                              activation=tf.nn.relu
                                              )
        self.__pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.__conv3 = tf.keras.layers.Conv2D(filters=64,
                                              kernel_size=[5, 5],
                                              padding="same",
                                              activation=tf.nn.relu
                                              )
        self.__pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.__flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.__full_connect1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.__full_connect2 = tf.keras.layers.Dense(units=10)

    def __call__(self, inputs):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        outputs = self.__conv1(inputs)
        outputs = self.__pool2(outputs)
        outputs = self.__conv3(outputs)
        outputs = self.__pool4(outputs)
        outputs = self.__flatten(outputs)
        outputs = self.__full_connect1(outputs)
        outputs = self.__full_connect2(outputs)

        return outputs

    def predict(self, inputs):
        outputs = self(inputs)

        return tf.argmax(outputs, axis=-1)


class MyNN:
    def __init__(self, fl=100, sl=10, lr=0.001, nb=10000, batch=50, show_steps=50, dl=DataLoader()):
        self.__learning_rate = lr
        self.__num_batches = nb
        self.__batch = batch
        self.data_loader = dl
        self.my_nn = MyCNN()# MLP(first_layer=fl, second_layer=sl)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate)
        self.__cost_vals = []
        self.__iters = []
        self.__show_steps = show_steps
        self.__took_time = 0.0
        self.__check_point = tf.train.Checkpoint(myAwsomeModel=self.my_nn)
        self.__training_recorder = tf.contrib.summary.create_file_writer('./tensorboard')
        self.__is_new_training = False

    def __str__(self):
        return "Learning tensorflow~"
    __repr__ = __str__

    def __call__(self, inputs):
        return self.predict(inputs)

    def model_test(self):
        num_eval_samples = np.shape(self.data_loader.eval_labels)[0]
        y_pred = self.my_nn.predict(self.data_loader.eval_data).numpy()
        print("test accuracy: {0}%".format(sum(y_pred == self.data_loader.eval_labels) / num_eval_samples * 100))

    def plot_cost_curve(self):
        plt.plot(self.__iters, self.__cost_vals, 'r', label='Lose function curve')
        plt.legend()
        plt.show()

    def train(self):
        print('Training start!')

        begin = time.time()

        with self.__training_recorder.as_default(), tf.contrib.summary.always_record_summaries():
            cur_cost = 0.0
            record_step = 1
            for i in range(self.__num_batches):
                data_set = self.data_loader.get_batch(self.__batch)

                with tf.GradientTape() as tape:
                    y_predict = self.my_nn(tf.convert_to_tensor(data_set[0]))
                    cost = tf.losses.sparse_softmax_cross_entropy(labels=data_set[1], logits=y_predict)

                    if (i+1) % self.__show_steps == 0:
                        tf.contrib.summary.scalar("loss", cost, step=record_step)
                        record_step += 1

                grads = tape.gradient(cost, self.my_nn.variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.my_nn.variables))

                '''if (i + 1) % self.__show_steps == 0:
                    cur_cost /= float(self.__show_steps)
                    cur_iter = (i + 1) // self.__show_steps
                    print("Iter: {0}, Cost: {1}".format(cur_iter, cur_cost))

                    self.__cost_vals.append(cur_cost)
                    self.__iters.append(cur_iter)

                    cur_cost = 0
                else:
                    cur_cost += cost.numpy()'''

        print('Training finished!')
        self.__is_new_training = True

        # The time training has taken
        end = time.time()
        print('Training takes {0}s'.format(end - begin))
        # Evaluate model
        self.model_test()
        # Plot cost curve
        # self.plot_cost_curve()
        # Save model
        self.save_model()

    def predict(self, inputs):
        return self.my_nn.predict(inputs)

    def save_model(self):
        if not self.__is_new_training:
            print('No new model to save!')
            return

        print('Saving Neural Network...')
        ID = int(time.time() % 1000000000)
        self.__check_point.save('./my_nn_models/model_' + str(ID) + '.ckpt')
        print("Finished!")

    def restore_model(self, path='./my_nn_models'):
        self.__check_point.restore(tf.train.latest_checkpoint(path))





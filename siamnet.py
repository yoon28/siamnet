import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from omniLoader import OmniglotLoader as og

class SiameseNet():
    
    eps = 1e-10
    learn_rate = 1e-6

    x_1 = tf.placeholder(tf.float32, shape=[None, og.im_size, og.im_size, og.im_channel])
    x_2 = tf.placeholder(tf.float32, shape=[None, og.im_size, og.im_size, og.im_channel])
    y = tf.placeholder(tf.float32, shape=[None])
    model_param = {'filter_size':(10, 7, 4), 'feature_size':(64, 128, 128), 'n_stack':3}

    def build_model(self, inputs, varscope, resuing=False):
        param = self.model_param
        input_channel = og.im_channel
        layer = inputs
        for l in range(param['n_stack']):
            with tf.variable_scope('{}_conv_{}'.format(varscope, l), reuse=resuing):
                filter_sz = param['filter_size'][l]
                feature_sz = param['feature_size'][l]
                filters = tf.get_variable('filter_{}'.format(l), [filter_sz, filter_sz, input_channel, feature_sz],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable('bias_{}'.format(l), [feature_sz], initializer=tf.constant_initializer(0.0))
                Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='VALID')
                activ = tf.nn.relu(Z + bias)
                layer = tf.nn.max_pool(activ, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
                input_channel = feature_sz
        last_layer = param['n_stack']
        with tf.variable_scope('{}_conv_{}'.format(varscope, last_layer), reuse=resuing):
            filters = tf.get_variable('filter_{}'.format(last_layer), [4, 4, param['feature_size'][last_layer-1], 256],
                initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('bias_{}'.format(last_layer), [256], initializer=tf.constant_initializer(0.0))
            Z = tf.nn.conv2d(layer, filters, strides=[1,1,1,1], padding='VALID')
            activ = tf.nn.relu(Z + bias)
            layer = tf.contrib.layers.flatten(activ)
        
        with tf.variable_scope('{}_dense'.format(varscope), reuse=resuing):
            output = tf.layers.dense(layer, 4096, activation=tf.nn.sigmoid, use_bias=True,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='fc_layer', reuse=resuing)
        
        return output
    
    def __init__(self, sharing=True):

        varscope='x' if sharing else 'x1'
        self.h1 = self.build_model(self.x_1, varscope, False)
        varscope='x' if sharing else 'x2'
        self.h2 = self.build_model(self.x_2, varscope, sharing)

        l1dist = tf.abs(self.h1-self.h2)
        with tf.variable_scope('dist_comp'):
            self.alpha = tf.get_variable('alpha', [1, 4096],
                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.2))
        
        self.logits = tf.reduce_sum(tf.multiply(l1dist, self.alpha), 1, keepdims=True)
        ys = tf.expand_dims(self.y, 1)
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=self.logits))
        optim = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        grad = optim.compute_gradients(self.loss)
        self.training = optim.apply_gradients(grad)

        correct = tf.to_float(tf.greater(tf.squeeze(self.logits, axis=[1]), 0.5))
        self.accuracy = tf.reduce_sum(correct)/tf.to_float(tf.shape(self.y)[0])

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    loader = og(0)
    model = SiameseNet(sharing=True)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    step = 0
    while True:
        x1, x2, y = loader.getTrainSample()
        [_, h1, h2, logits, loss, accuracy] = session.run([model.training, model.h1, model.h2, model.logits, model.loss, model.accuracy],
            feed_dict={model.x_1:x1, model.x_2:x2, model.y:y})

        step += 1
        if (step %50) == 0:
            print('{}\n  train: loss={}, acc={:.2%}'.format(step, loss, accuracy))
            x1_t, x2_t, y_t = loader.getTestSample(batch_sz=128)
            [acc_t] = session.run([model.accuracy], feed_dict={model.x_1:x1_t, model.x_2:x2_t, model.y:y_t})
            print('  test: acc={:.2%}'.format(acc_t))
    
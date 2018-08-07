import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from omniLoader import OmniglotLoader as og

class SiameseNet():
    
    eps = 1e-10
    learn_rate = 1e-4

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
        
        with tf.variable_scope('loss_and_training'):
            self.logits = tf.reduce_sum(tf.multiply(l1dist, self.alpha), 1, keepdims=True)
            ys = tf.expand_dims(self.y, 1)
            self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=ys, logits=self.logits))
            optim = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            grad = optim.compute_gradients(self.loss)
            self.training = optim.apply_gradients(grad)

        with tf.variable_scope('accuracy'):
            self.n_trues = tf.count_nonzero(self.y, dtype=tf.float32)
            self.positives = tf.greater(tf.squeeze(tf.sigmoid(self.logits), axis=[1]), 0.5)
            self.n_pos = tf.count_nonzero(self.positives, dtype=tf.float32)
            self.tp = tf.logical_and(self.positives, tf.equal(self.y, 1.0))
            self.n_tp = tf.count_nonzero(self.tp, dtype=tf.float32)
            self.n_falses = tf.count_nonzero(tf.equal(self.y, 0.0), dtype=tf.float32)
            self.fp = tf.logical_and(self.positives, tf.equal(self.y, 0.0))
            self.n_fp = tf.count_nonzero(self.fp, dtype=tf.float32)

            def outputzero() : return 0.0
            def comp_pr(): return tf.div(self.n_tp, self.n_pos)
            def comp_rcll(): return tf.div(self.n_tp, self.n_trues)
            def comp_fpr(): return tf.div(self.n_fp, self.n_falses)

            self.precision = tf.cond(tf.greater(self.n_pos, 0.0), true_fn=comp_pr, false_fn=outputzero)
            self.recall = tf.cond(tf.greater(self.n_trues, 0.0), true_fn=comp_rcll, false_fn=outputzero)
            self.fpr = tf.cond(tf.greater(self.n_falses, 0.0), true_fn=comp_fpr, false_fn=outputzero)
            
        with tf.variable_scope('20_way_oneshot_classification'):
            self.pred = tf.squeeze(tf.sigmoid(self.logits), axis=[1])
            self.pred = tf.expand_dims(self.pred, 0)
            self.loc = tf.squeeze(tf.where(tf.equal(self.y, 1.0)), axis=[1])
            self.correct = tf.squeeze(tf.nn.in_top_k(self.pred, self.loc, k=1))

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
        [_, h1, h2, logits, loss, precision, recall, fpr] = session.run([model.training, model.h1, model.h2, model.logits, model.loss, model.precision, model.recall, model.fpr],
            feed_dict={model.x_1:x1, model.x_2:x2, model.y:y})

        step += 1
        if (step %50) == 0:
            print('{}\n train: loss={:.5f}, lr={}, pr={:.2%}, rcll={:.2%}, fpr={:.2%}'.format(step, loss, model.learn_rate, precision, recall, fpr))
            x1_t, x2_t, y_t = loader.getTestSample(batch_sz=128)
            [pr_t, rcll_t, fpr_t] = session.run([model.precision, model.recall, model.fpr], feed_dict={model.x_1:x1_t, model.x_2:x2_t, model.y:y_t})
            print(' test: pr={:.2%}, rcll={:.2%}, fpr={:.2%}'.format(pr_t, rcll_t, fpr_t))

            acc_oneshot = []
            n_try = 150
            for i in range(n_try):
                x1_20, x2_20, y_20 = loader.getTestSample_20way()
                [pred_20, loc_20, corr_20] = session.run([model.pred, model.loc, model.correct],
                    feed_dict={model.x_1: x1_20, model.x_2: x2_20, model.y: y_20})
                acc_oneshot.append(corr_20)
            acc_20 = float(sum(acc_oneshot))/float(n_try)
            print(' 20-way: acc={:.2%}'.format(acc_20))

        # if not (np.sum(np.float32(np.float32(np.squeeze(1/(1+np.exp(-logits)))>0.5) == correct)) == y.shape[0]):
        #     print('error 1')
        if (step % 500) == 0 and model.learn_rate > 1e-6:
            model.learn_rate *= 0.98

        # sanity check code
        if (step % 1000) == 0:
            ck_n_trues = np.sum(np.float32(y == 1))
            ck_n_false = np.sum(np.float32(y == 0))
            ck_pred = np.squeeze(1/(1+np.exp(-logits)))
            ck_pos = ck_pred > 0.5
            ck_n_pos = np.sum(np.float32(ck_pos))
            ck_tp = np.logical_and(ck_pos, (y==1))
            ck_fp = np.logical_and(ck_pos, (y==0))
            ck_prc = np.sum(ck_tp)/ck_n_pos if ck_n_pos>0.0 else 0.0
            ck_rcll = np.sum(ck_tp)/ck_n_trues if ck_n_trues>0.0 else 0.0
            ck_fpr = np.sum(ck_fp)/ck_n_false if ck_n_false>0.0 else 0.0

            if np.abs(ck_prc-precision) > 1e-5:
                print('error precision')
            if np.abs(ck_rcll-recall) > 1e-5:
                print('error recall')
            if np.abs(ck_fpr-fpr) > 1e-5:
                print('error fpr')

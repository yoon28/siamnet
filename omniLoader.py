
import os
import cv2
import numpy as np

class OmniglotLoader():
    
    dataloc = 'omniglot/'
    backset = dataloc + 'images_background/'
    evalset = dataloc + 'images_evaluation/'
    
    n_examples_per_char = 20
    langs1 = os.listdir(backset)
    langs2 = os.listdir(evalset)
    
    train_chars = {}
    test_chars = {}
    for lang in langs1:
        for char in os.listdir(backset+lang):
            files = os.listdir(backset+lang+'/'+char)
            class_id = int(files[0].split('_')[0])
            if class_id <= 1200:
                train_chars[class_id] = backset+lang+'/'+char
            else:
                test_chars[class_id] = backset+lang+'/'+char

    for lang in langs2:
        for char in os.listdir(evalset+lang):
            files = os.listdir(evalset+lang+'/'+char)
            class_id = int(files[0].split('_')[0])
            if class_id <= 1200:
                train_chars[class_id] = evalset+lang+'/'+char
            else:
                test_chars[class_id] = evalset+lang+'/'+char
    
    n_class_train = len(train_chars) 
    n_class_test = len(test_chars)
    im_size = 105
    im_channel = 1
    
    def __init__(self, rdseed, batch_sz=128):
        np.random.seed(rdseed)
        self.batch_size = batch_sz

    def getTrainSample(self, disp=False):
        batch_size = self.batch_size
        x_i_1 = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel], dtype=np.float32) # one channel (the last dimension)
        x_i_2 = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel], dtype=np.float32)
        y_i = np.zeros([batch_size], dtype=np.float32)
        
        classes = list(self.train_chars.keys())
        batch_idx = np.random.permutation(batch_size)
        for b in range(batch_size):
            issame = np.random.randint(2)
            y_i[batch_idx[b]] = issame
            if(issame):
                cl_sp = np.random.choice(classes)
                clss = self.train_chars[cl_sp]
                drawers = np.random.choice(self.n_examples_per_char, 2, replace=False) +1
                x_1 = clss + '/{:04d}_{:02d}.png'.format(cl_sp, drawers[0])
                x_2 = clss + '/{:04d}_{:02d}.png'.format(cl_sp, drawers[1])
                img = cv2.imread(x_1, -1)
                x_i_1[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
                img = cv2.imread(x_2, -1)
                x_i_2[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
            else:
                cl_sp = np.random.choice(classes, 2, replace=False)
                clss_1 = self.train_chars[cl_sp[0]]
                drawer_1 = np.random.randint(self.n_examples_per_char)+1
                x_1 = clss_1 + '/{:04d}_{:02d}.png'.format(cl_sp[0], drawer_1)
                clss_2 = self.train_chars[cl_sp[1]]
                x_2 = clss_2 + '/{:04d}_{:02d}.png'.format(cl_sp[1], np.random.randint(self.n_examples_per_char)+1)
                img = cv2.imread(x_1, -1)
                x_i_1[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
                img = cv2.imread(x_2, -1)
                x_i_2[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1

        if disp: self.displayImage(x_i_1, x_i_2, y_i)
        return x_i_1, x_i_2, y_i

    def getTestSample(self, batch_sz=32, disp=False):
        batch_size = batch_sz
        x_i_1 = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel], dtype=np.float32) # one channel (the last dimension)
        x_i_2 = np.zeros([batch_size, self.im_size, self.im_size, self.im_channel], dtype=np.float32)
        y_i = np.zeros([batch_size], dtype=np.float32)
        
        classes = list(self.test_chars.keys())
        batch_idx = np.random.permutation(batch_size)
        for b in range(batch_size):
            issame = np.random.randint(2)
            y_i[batch_idx[b]] = issame
            if(issame):
                cl_sp = np.random.choice(classes)
                clss = self.test_chars[cl_sp]
                drawers = np.random.choice(self.n_examples_per_char, 2, replace=False) +1
                x_1 = clss + '/{:04d}_{:02d}.png'.format(cl_sp, drawers[0])
                x_2 = clss + '/{:04d}_{:02d}.png'.format(cl_sp, drawers[1])
                img = cv2.imread(x_1, -1)
                x_i_1[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
                img = cv2.imread(x_2, -1)
                x_i_2[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
            else:
                cl_sp = np.random.choice(classes, 2, replace=False)
                clss_1 = self.test_chars[cl_sp[0]]
                drawer_1 = np.random.randint(self.n_examples_per_char)+1
                x_1 = clss_1 + '/{:04d}_{:02d}.png'.format(cl_sp[0], drawer_1)
                clss_2 = self.test_chars[cl_sp[1]]
                x_2 = clss_2 + '/{:04d}_{:02d}.png'.format(cl_sp[1], np.random.randint(self.n_examples_per_char)+1)
                img = cv2.imread(x_1, -1)
                x_i_1[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1
                img = cv2.imread(x_2, -1)
                x_i_2[batch_idx[b], :, :, 0] = img.astype(np.float)/127.5 -1

        if disp: self.displayImage(x_i_1, x_i_2, y_i)
        return x_i_1, x_i_2, y_i

    def displayImage(self, x_i, x_h, y_i):
        sz = x_i.shape[0]
        for s in range(sz):
            label = int(y_i[s])
            cv2.namedWindow('x1_{}'.format(label), cv2.WINDOW_NORMAL)
            cv2.imshow('x1_{}'.format(label), x_i[s,:,:,:])
            cv2.namedWindow('x2_{}'.format(label), cv2.WINDOW_NORMAL)
            cv2.imshow('x2_{}'.format(label), x_h[s,:,:,:])
            cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    loader = OmniglotLoader(0)
    step = 0
    while True:
        x_1, x_2, y_hat = loader.getTrainSample(False)
        # x_1, x_2, y_hat = loader.getTestSample(disp=True)
        print(step)
        step += 1
        
    print(loader.train_chars)
    print(loader.test_chars)
    
import caffe
import numpy as np
from timer import Timer
from collections import defaultdict

SEMI_HARD = True
margin = 0.2

class TripletSampleLayer(caffe.Layer):

    def setup(self, bottom, top):
        """Setup the TripletSampleLayer."""
        top[0].reshape(*bottom[0].data.shape)
        top[1].reshape(*bottom[0].data.shape)
        top[2].reshape(*bottom[0].data.shape)
        self._timer = Timer()
        self._negative_timer = Timer()

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""

        bottom_data = np.array(bottom[0].data)
        bottom_label = np.array(bottom[1].data)
        self.index_map = []

        top_anchor = []
        top_positive = []
        top_negative = []

        label_index_map = defaultdict(list)
        for i in xrange(bottom[0].num):
            label_index_map[bottom_label[i]].append(i)

        for i in xrange(bottom[0].num):
            anchor_label = bottom_label[i]
            anchor = bottom_data[i]

            positive = None
            positive_index = None
            top_ap = 0
            for j in range(len(label_index_map[anchor_label])):
                current_positive_index = label_index_map[anchor_label][j]
                if len(label_index_map[anchor_label]) > 1 and current_positive_index == i:
                    continue
                current_positive = bottom_data[current_positive_index]
                current_ap = np.sum((anchor - current_positive) ** 2)
                if current_ap > top_ap:
                    top_ap = current_ap
                    positive = current_positive
                    positive_index = current_positive_index



            negative_label = anchor_label
            negative_index = np.random.choice(
                label_index_map[negative_label])
            negative = bottom_data[negative_index]
            # still need semi-hard mining?
            # Yes we need. Setting it as True is for entering the "while" below at the beginning.
            semihard = True
            # max iteration
            # this "iteration" means how many times we randomly select the negative.
            max_iter = bottom[0].num * 2
            while len(label_index_map) > 1 and (negative_label == anchor_label or semihard):
                negative_label = np.random.choice(label_index_map.keys())
                negative_index = np.random.choice(
                    label_index_map[negative_label])
                negative = bottom_data[negative_index]

                #Note:here to decide whether to use semihard
                if SEMI_HARD:
                    ap = np.sum((anchor - positive) ** 2)
                    an = np.sum((anchor - negative) ** 2)
                    #semihard = not((ap + margin - an > 0) and (ap < an))#this condition is for the first 20000 iterations
                    semihard = (ap <= an)
                else:
                    semihard = False

                max_iter -= 1
                if max_iter <= 0:
                    #print 'Semi-hard failed'
                    negative_label = anchor_label
                    while negative_label == anchor_label:
                        negative_label = np.random.choice(label_index_map.keys())
                        negative_index = np.random.choice(label_index_map[negative_label])
                        negative = bottom_data[negative_index]
                    break
            
            #print [anchor_label, negative_label]
            #print [i, positive_index, negative_index]

            top_anchor.append(anchor)
            top_positive.append(positive)
            top_negative.append(negative)

            self.index_map.append([i, positive_index, negative_index])

        top[0].reshape(*np.array(top_anchor).shape)
        top[1].reshape(*np.array(top_anchor).shape)
        top[2].reshape(*np.array(top_anchor).shape)
        top[0].data[...] = np.array(top_anchor)
        top[1].data[...] = np.array(top_positive)
        top[2].data[...] = np.array(top_negative)

        # self._timer.toc()
        #
        # print 'Sample:', self._timer.average_time,
        # self._negative_timer.average_time

    def backward(self, top, propagate_down, bottom):
        """Get top diff and compute diff in bottom."""

        if propagate_down[0]:
            bottom_diff = np.zeros(top[0].diff.shape)

            for i in xrange(top[0].num):
                bottom_diff[self.index_map[i][0]] += top[0].diff[i]
                bottom_diff[self.index_map[i][1]] += top[1].diff[i]
                bottom_diff[self.index_map[i][2]] += top[2].diff[i]

            bottom[0].diff[...] = bottom_diff

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

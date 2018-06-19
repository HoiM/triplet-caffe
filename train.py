#!/usr/bin/env python

import os
import sys
import google.protobuf as pb2
import matplotlib.pyplot as plt
CAFFE_ROOT = "/path/to/caffe/"
sys.path.append(CAFFE_ROOT + "python/")

import caffe
import triplet
from timer import Timer
from caffe.proto import caffe_pb2
import numpy as np

class SolverWapper(object):
    
    def __init__(self, solver, output_dir, pretrained_model=None, gpu_id=0):
        
        self.output_dir = output_dir
        
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        self.solver = caffe.SGDSolver(solver)
        if pretrained_model is not None:
            print "Loading pretrained model, weights from {:s}".format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)
        
        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver, "rt") as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

    def train_model(self, plot_iter):
        
        display = self.solver_param.display
        snapshot = self.solver_param.snapshot
        max_iters = self.solver_param.max_iter
 
        last_snapshot_iter = -1
        timer = Timer()
        losstxt = os.path.join(self.output_dir, 'loss.txt')
        f = open(losstxt, 'w')

        loss_list = []

        while self.solver.iter < max_iters:
        

            timer.tic()
            self.solver.step(1)
            timer.toc()

            loss = self.solver.net.blobs['loss'].data[0]
            loss_list.append(loss)
            f.write('{} {}\n'.format(self.solver.iter - 1, loss))
            f.flush()


            if self.solver.iter % (1 * display) == 0:
                print '---------------------------------------------------------'
                print 'speed: {:.3f}s / iter'.format(timer.average_time)
                print 'time remains: {}s'.format(timer.remain(self.solver.iter, max_iters))
                print '---------------------------------------------------------'

            if self.solver.iter % plot_iter == 0:
                x = np.linspace(self.solver.iter - len(loss_list) + 1, self.solver.iter, len(loss_list))
                y = loss_list
                plt.plot(x,y)
                plt.ylabel("loss")
                plt.xlabel("iters")
                title = "iter:" + str(self.solver.iter - plot_iter) + "~" + str(self.solver.iter)
                plt.title(title)
                plotpath = output_dir + "loss_" + "iter_" +  str(self.solver.iter - plot_iter) +  "_" + str(self.solver.iter) + ".png"
                plt.savefig(plotpath)
                print '---------------------------------------------'
                print 'loss saved to: ' + plotpath
                print '---------------------------------------------'
                plt.close()
                loss_list = []
        f.close()


if __name__ == '__main__':
    solver = '/path/to/solver.prototxt'
    output_dir = '/path/to/output/'
    pretrained_model = '/path/to/pretrained.caffemodel'
    gpu_id = 0
    plot_iter = 1000000 # plot and save the loss after some iterations
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    sw = SolverWapper(solver, output_dir, pretrained_model, gpu_id)
    
    print "Solving..."
    sw.train_model(plot_iter)
    print "Solving done..."


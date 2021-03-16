# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as scio
import scipy
from PIL import Image

import time
import random
import os

from VGG_backbone import VGG_10

import tools
import relabel_tools
import data_loader_ShanghaiTech as Shanghai




print("-----------------------------------------------------------------------------\n")



###
# model setting
batch_SIZE = 1
Epoch = 0

first_write_flag = False
first_write_epoch = 30
rewrite_epoch = 5
scale = np.array([5])



train_procedure_Shanghai = [("part_B_final","train_data")]
test_procedure_Shanghai = [("part_B_final","test_data")]




def common_conv2d(z , in_filter = None , out_filter = None ,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [1,1,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.compat.v1.nn.conv2d(z , W , strides=[1,1,1,1] ,  padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z
    
    
def dilated_conv2d(z , in_filter = None , out_filter = None , dilated_rate = None,Name = None):

    with tf.variable_scope(Name):    
        W = tf.compat.v1.get_variable(name = Name+"_W" , shape = [3,3,in_filter,out_filter])
        b = tf.compat.v1.get_variable(name = Name+"_b" , shape = [out_filter],initializer = tf.compat.v1.zeros_initializer())
    
        z = tf.nn.atrous_conv2d(z , W , rate = dilated_rate , padding="SAME") + b
        z = tf.nn.relu(z)    
        
        return z

    


        

'''
--------------------------------------------------------------------------------------------------------------------------        

'''

def Training(fold = None , percentage = None):
    


    tf.compat.v1.reset_default_graph()


    x = tf.compat.v1.placeholder(dtype = tf.float32 ,  shape = [None , None , None , 3])
    GT = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , None , None , 3])
    Count = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [None , 1 ])
    
    LR = tf.compat.v1.placeholder(tf.float32)
    
    score_scale = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [1])

    look_cost = tf.compat.v1.placeholder(tf.float32)
    look_MAE = tf.compat.v1.placeholder(tf.float32)
    look_MSE = tf.compat.v1.placeholder(tf.float32)

    with tf.compat.v1.variable_scope("MODEL"):

        VGG = VGG_10()  
        z = VGG.forward(x)

        z = tf.compat.v1.image.resize_bilinear(z , [ 2 * tf.shape(z)[1] , 2 * tf.shape(z)[2]])        
        
        z = dilated_conv2d(z , 512 , 256 , 1 , "128")
        z = dilated_conv2d(z , 256 , 128 , 1 , "64")

        z = tf.layers.conv2d(z , 1 , [1,1] , strides=[1,1] , padding="SAME")

        z = tf.abs(z)
        

    loss , decoupled_density , expectation , variance = tools.Re_estimate(z , GT , score_scale)


    all_variable = tf.compat.v1.trainable_variables()    
    L2_reg = 5e-6 * tf.reduce_sum([ tf.nn.l2_loss(v) for v in all_variable ])

    
    performance = tools.compute_MAE_and_MSE(z , Count)


    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LR).minimize(loss , name = "Adam_All")
    weight_decay_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1.0).minimize(L2_reg , name = "Weight_decay")

    tf.compat.v1.summary.scalar("loss", look_cost)
    tf.compat.v1.summary.scalar("MAE" , look_MAE)
    tf.compat.v1.summary.scalar("MSE" , look_MSE)

    
    merged = tf.compat.v1.summary.merge_all()
    
    Saver = tf.compat.v1.train.Saver(max_to_keep = 20)
    
    initial = tf.compat.v1.global_variables_initializer()


    with tf.compat.v1.Session() as sess:

        print("-----------------------------------------------------------------------------\n")

        sess.graph.finalize()
        
        sess.run(initial)
        
        Saver.restore(sess , ".\\model_last_epoch_for_inference_{0}\\BLpp.ckpt".format(fold))
        
        print("\nModel restoring...\n")

        global first_write_flag

        for epoch in range(Epoch+1):
                         

            for (A_or_B ,  train_dir) in train_procedure_Shanghai:

                seed = 860521
                
                random.seed(seed)
                
                data_list = Shanghai.get_data_list(A_or_B = A_or_B , train_or_test = train_dir)

                data_list = data_list[fold * int(percentage * len(data_list)):(fold+1) * int(percentage * len(data_list))]

                     
 
            # rewrite            
            if True:
                
                first_write_flag = True
                print("[!] Re-estimate writing...")

                C = 0
                var_list = []
                for file in data_list:

                    (X_train_batch , Y_train_batch) = Shanghai.data_loader_pipeline(A_or_B = A_or_B , train_or_test = train_dir , 
                                                                                    data_list = [file] , write = True)
                
                    

                    dens , expects , var = sess.run([decoupled_density, expectation , variance],feed_dict={x:X_train_batch , 
                                                                                          GT:Y_train_batch[0][0] ,
                                                                                          Count:Y_train_batch[0][1],
                                                                                          score_scale:scale})

                    var_list.append(var[0])
               


                    c = relabel_tools.rewrite_variance(A_or_B , file , dens , expects , var , True , Y_train_batch[0][0])
                    C += c
                
                var_array = np.concatenate(var_list , axis = 0)
                print(var_array.shape , C)                
                print("[!] Re-estimate writing finished.")
                
                return var_array


if __name__ == "__main__" :
    

    percentage = 0.2
   
    for i in range(1):
        
        var = Training(fold = i , percentage = percentage)

        a = 0.5*(var[:,0]+var[:,1])
        a = np.sqrt(a)
        print(np.max(a) , np.min(a))
    

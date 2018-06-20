
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


def weight_variable(shape):
     
    w = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(w)

def bias_variable(shape):
     
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

 
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
 
    shape = [filter_size, filter_size, num_input_channels, num_filters]
 
    weights = weight_variable(shape)
    biases = bias_variable([num_filters])
 
    
    layer = tf.nn.relu(tf.nn.conv2d(input=input,
                                    filter=weights,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME') + biases)

    if use_pooling: 
        return max_pool_2x2(layer), weights

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 
    weights = weight_variable([num_inputs, num_outputs])
    biases = bias_variable([num_outputs])
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
 
    return layer

 

 
x = tf.placeholder(tf.float32, shape=[None, 28*28], name='input_data')
x_image = tf.reshape(x, [-1,28,28,1])
 
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='correct_labels')

# fist conv layer
convlayer1, w1 = new_conv_layer(x_image, 1, 5, 32)
# second conv layer
convlayer2, w2 = new_conv_layer(convlayer1, 32, 5, 64)
# flat layer
flat_layer, num_features = flatten_layer(convlayer2)
# fully connected layer
fclayer = new_fc_layer(flat_layer, num_features, 1024)
# DROPOUT
keep_prob = tf.placeholder(tf.float32)
drop_layer = tf.nn.dropout(fclayer, keep_prob)
# final layer
W_f = weight_variable([1024, 10])
b_f = bias_variable([10])
y_f = tf.matmul(drop_layer, W_f) + b_f
y_f_softmax = tf.nn.softmax(y_f)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_f))

# train step
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(y_f_softmax, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init
init = tf.global_variables_initializer()


# In[4]:


num_steps =1000
batch_size = 16
test_size = 10000
saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(init)
    
    for step in range(num_steps):
        batch = mnist.train.next_batch(batch_size)
        
        if step % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %f' %(step, train_accuracy))
            save_path = saver.save(sess, "/home/ashutosh/Documents/tens/model2.ckpt",global_step=step)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print 'Done!'
    print 'Evaluating...'
    
     
    file_writer = tf.summary.FileWriter('/home/ashutosh/Documents/tens', sess.graph)
    file_writer.add_graph(sess.graph)


import tensorflow as tf
import matplotlib.pyplot as plt
import os, nltk
from miscc.config import cfg
import numpy as np

def fc(inputs, num_out, name, activation_fn=None, biased=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    return tf.layers.dense(inputs=inputs, units=num_out, activation=activation_fn, kernel_initializer=w_init, use_bias=biased, name=name)

def concat(inputs, axis, name):
    return tf.concat(values=inputs, axis=axis, name=name)

def batch_normalization(inputs, is_training, name, activation_fn=None):
    output = tf.layers.batch_normalization(
                    inputs,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=is_training,
                    name=name
                )
    if activation_fn is not None:
        output = activation_fn(output)
    return output

def reshape(inputs, shape, name):
    return tf.reshape(inputs, shape, name)

def Conv2d(input, k_h, k_w, c_o, s_h, s_w, name, activation_fn=None, padding='VALID', biased=False):
    c_i = input.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(input, kernel)
        if biased:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)
        return output

def add(inputs, name):
    return tf.add_n(inputs, name=name)

def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)

def flatten(input, name):
    input_shape = input.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        input = tf.reshape(input, [-1, dim])
    return input

#############################################################################################################
# Other Layers
#############################################################################################################
weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None
weight_regularizer_fc = None

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample(x):
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])
    return gap

def global_sum_pooling(x) :
    gsp = tf.reduce_sum(x, axis=[1, 2])
    return gsp

def max_pooling(x) :
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2, padding='SAME')

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)

def tanh(x):
    return tf.tanh(x)

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-05, center=True, scale=True,
                                        updates_collections=None, is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)
    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)
    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left
            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                               regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        else:
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)
        return x

def deconv(x, channels, kernel=3, stride=1, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()
        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]
        else:
            output_shape = [x_shape[0], x_shape[1] * stride + max(kernel - stride, 0),
                           x_shape[2] * stride + max(kernel - stride, 0), channels]
        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init,
                                                regularizer=weight_regularizer)
            x = tf.nn.convd_transpose(x, filter=spectral_norm(w), output_shape=output_shape,
                                     strides=[1, stride, stride, 1], padding=padding)
        else:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init,
                                          kernel_regularizer=weight_regularizer, strides=stride, padding=padding, use_bias=use_bias)
        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x, 'fc/flatten')
        shape = x.get_shape().as_list()
        channels = shape[-1]
        if sn:
            w = tf.get_variable("kernel", [channels, units], dtype=tf.float32, initializer=weight_init, regularizer=weight_regularizer_fc)
            if use_bias:
                bias = tf.get_variable("bias", [units], initializer=tf.constant_initializer(0.0))
                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))
        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer_fc, use_bias=use_bias)
        return x

def up_resblock(x_init, channels, use_bias=True, is_training=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = batch_norm(x_init, is_training)
            x = relu(x)
            x = up_sample(x, scale_factor=2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=False, sn=sn)
        with tf.variable_scope('res2'):
            x = batch_norm(x, is_training)
            x = relu(x)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        with tf.variable_scope('shorcut'):
            x_init = up_sample(x_init, scale_factor=2)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)
        return x + x_init

def down_resblock(x_init, channels, to_down=True, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        init_channel = x_init.shape.as_list()[-1]
        with tf.variable_scope('res1'):
            x = lrelu(x_init, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
        with tf.variable_scope('res2'):
            x = lrelu(x, 0.2)
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            if to_down:
                x = down_sample(x)
        if to_down or init_channel != channels:
            with tf.variable_scope('shortcut'):
                x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)
                if to_down:
                    x_init = down_sample(x_init)
        return x + x_init

def init_down_resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = lrelu(x, 0.2)
        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = down_sample(x)
        with tf.variable_scope('shortcut'):
            x_init = down_sample(x_init)
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, sn=sn)
        return x + x_init

#################################################
# DO NOT CHANGE 
class GENERATOR:
#################################################
    def __init__(self, input_z, input_rnn, is_training=False, reuse=False):
        self.input_z = input_z
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128
        self.gf_dim = 128
        self.image_size = 64
        self.c_dim = 3
        self.layer_num = 3
        self.sn = True
        self._build_model()
    
    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        gf_dim = self.gf_dim
        t_dim = self.t_dim
        c_dim = self.c_dim
        with tf.variable_scope("generator", reuse=self.reuse):
            net_txt = fc(inputs=self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='rnn_fc')
            net_in = concat([self.input_z, net_txt], axis=1, name='concat_z_txt')
            net_h0 = fc(inputs=net_in, num_out=gf_dim*8*s16*s16, name='g_h0/fc', biased=False)
            net_h0 = batch_normalization(net_h0, activation_fn=None, is_training=self.is_training, name='g_h0/batch_norm')
            net_h0 = reshape(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
            ch = 1024
            x = fully_connected(net_h0, units=4*4*ch, sn=self.sn, scope='fc')
            x = tf.reshape(x, [-1, 4, 4, ch])
            x = up_resblock(x, channels=ch, is_training=self.is_training, sn=self.sn, scope='front_resblock_0')
            for i in range(self.layer_num // 2) :
                x = up_resblock(x, channels=ch // 2, is_training=self.is_training, sn=self.sn, scope='middle_resblock_' + str(i))
                ch = ch // 2
            x = self.google_attention(x, channels=ch, scope='self_attention')
            for i in range(self.layer_num // 2, self.layer_num) :
                x = up_resblock(x, channels=ch // 2, is_training=self.is_training, sn=self.sn, scope='back_resblock_' + str(i))
                ch = ch // 2
            x = batch_norm(x, self.is_training)
            x = relu(x)
            x = conv(x, channels=self.c_dim, kernel=3, stride=1, pad=1, pad_type='reflect', scope='g_logit')
            self.outputs = tanh(x)
            self.logits = x
    
    def google_attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):
            batch_size, height, width, num_channels = x.get_shape().as_list()
            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            f = max_pooling(f)
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
            h = conv(x, channels // 2, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            h = max_pooling(h)
            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
            beta = tf.nn.softmax(s)  # attention map
            o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])  # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            x = gamma * o + x
        return x
                  
#################################################
# DO NOT CHANGE 
class DISCRIMINATOR:
#################################################
    def __init__(self, input_image, input_rnn, is_training=False, reuse=False):
        self.input_image = input_image
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self.image_size = 64
        self._build_model()
    
    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
        df_dim = self.df_dim
        t_dim = self.t_dim
        with tf.variable_scope("discriminator", reuse=self.reuse):
            net_h0 = Conv2d(self.input_image, 4, 4, df_dim, 2, 2, name='d_h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)
            
            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='d_h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h1/batchnorm')
            
            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='d_h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h2/batchnorm')
            
            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='d_h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=None, is_training=self.is_training, name='d_h3/batchnorm')
            
            net = Conv2d(net_h3, 1, 1, df_dim*2, 1, 1, name='d_h4_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm')
            net = Conv2d(net, 3, 3, df_dim*2, 1, 1, name='d_h4_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm2')
            net = Conv2d(net, 3, 3, df_dim*8, 1, 1, name='d_h4_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='d_h4_res/batchnorm3')
            
            net_h4 = add([net_h3, net], name='d_h4/add')
            net_h4_outputs = tf.nn.leaky_relu(net_h4)
            
            net_txt = fc(self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='d_reduce_txt/dense')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim1')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim2')
            net_txt = tf.tile(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            
            net_h4_concat = concat([net_h4_outputs, net_txt], axis=3, name='d_h3_concat')
            
            net_h4 = Conv2d(net_h4_concat, 1, 1, df_dim*8, 1, 1, name='d_h3/conv2d_2')
            net_h4 = batch_normalization(net_h4, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h3/batch_norm_2')
            
            net_ho = Conv2d(net_h4, s16, s16, 1, s16, s16, name='d_ho/conv2d', biased=True) # biased = True
            
            self.outputs = tf.nn.sigmoid(net_ho)
            self.logits = net_ho

#################################################
# DO NOT CHANGE 
class RNN_ENCODER:
#################################################
    def __init__(self, input_seqs, is_training=False, reuse=False):
        self.input_seqs = input_seqs
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128  
        self.rnn_hidden_size = 128
        self.vocab_size = 8000
        self.word_embedding_size = 256
        self.keep_prob = 1.0
        self.batch_size = 64
        self._build_model()
    
    def _build_model(self):
        w_init = tf.random_normal_initializer(stddev=0.02)
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
        with tf.variable_scope("rnnencoder", reuse=self.reuse):
            word_embed_matrix = tf.get_variable('rnn/wordembed', 
                shape=(self.vocab_size, self.word_embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.02),
                dtype=tf.float32)
            embedded_word_ids = tf.nn.embedding_lookup(word_embed_matrix, self.input_seqs)
            # RNN encoder
            LSTMCell = tf.contrib.rnn.BasicLSTMCell(self.t_dim, reuse=self.reuse)
            initial_state = LSTMCell.zero_state(self.batch_size, dtype=tf.float32)
            rnn_net = tf.nn.dynamic_rnn(cell=LSTMCell,
                                    inputs=embedded_word_ids,
                                    initial_state=initial_state,
                                    dtype=np.float32,
                                    time_major=False,
                                    scope='rnn/dynamic')
            self.rnn_net = rnn_net
            self.outputs = rnn_net[0][:, -1, :]

#################################################
# DO NOT CHANGE 
class CNN_ENCODER:
#################################################
    def __init__(self, inputs, is_training=False, reuse=False):
        self.inputs = inputs
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self._build_model()
    
    def _build_model(self):
        df_dim = self.df_dim
        with tf.variable_scope('cnnencoder', reuse=self.reuse):
            net_h0 = Conv2d(self.inputs, 4, 4, df_dim, 2, 2, name='cnnf/h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)
            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='cnnf/h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h1/batch_norm')
            
            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='cnnf/h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h2/batch_norm')
            
            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='cnnf/h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h3/batch_norm')
            
            net_h4 = flatten(net_h3, name='cnnf/h4/flatten')
            net_h4 = fc(net_h4, num_out=self.t_dim, name='cnnf/h4/embed', biased=False)
            
            self.outputs = net_h4
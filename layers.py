from inits import *
import tensorflow as tf
import scipy.io as sio
import numpy as np
import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=0):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse == True:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

    
class Layer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphElasticConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphElasticConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        self.alpha = 0.75
        self.gamma = 0.001
        self.k = placeholders['features'].shape[0].value
        self.I = tf.eye(self.k)
        self.n = 1./self.k

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        #-------Dropout-------
        x = tf.nn.dropout(inputs, 1-self.dropout)
        #-------Convolve-------   
        U = x
        zero_vec = tf.zeros_like(self.support)
        for i in range(4):
            # -------Updating M-------
            M = 1./(2*self.gamma)*self.support * dot(U, tf.transpose(U))                       
            for j in range(3):
                M1 = (self.n*self.I + tf.reduce_mean(M)*self.I - self.n*M)
                M2 = tf.reshape(tf.tile(tf.reduce_sum(M1,0),[self.k]),[self.k,self.k])
                M3 = tf.reshape(tf.tile(tf.reduce_sum(M,0),[self.k]),[self.k,self.k])
                M = M + M2 - self.n*M3
                M = tf.where(M > 0., M, zero_vec)
            # -------Updating U-------
            A = tf.where(M > 0.01, self.support, zero_vec)
            for g in range(3):
                U = self.alpha*dot(A, U)+(1.-self.alpha)*x
            
        output = dot(U, self.vars['weights'])
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)

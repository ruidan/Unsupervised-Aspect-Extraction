import keras.backend as K
from keras.engine.topology import Layer
from keras import initializations
from keras import regularizers
from keras import constraints
import numpy as np
import theano.tensor as T

class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) == list
        assert len(input_shape) == 2

        self.steps = input_shape[0][1]

        self.W = self.add_weight((input_shape[0][-1], input_shape[1][-1]),
                                    initializer=self.init,
                                    name='{}_W'.format(self.name),
                                    regularizer=self.W_regularizer,
                                    constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]

        y = K.transpose(K.dot(self.W, K.transpose(y)))
        y = K.expand_dims(y, dim=-2)
        y = K.repeat_elements(y, self.steps, axis=1)
        eij = K.sum(x*y, axis=-1)

        if self.bias:
            b = K.repeat_elements(self.b, self.steps, axis=0)
            eij += b

        eij = K.tanh(eij)
        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        return a

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        assert type(input_tensor) == list
        assert type(mask) == list

        x = input_tensor[0]
        a = input_tensor[1]

        a = K.expand_dims(a)
        weighted_input = x * a

        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

    def compute_mask(self, x, mask=None):
        return None

class WeightedAspectEmb(Layer):
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 weights=None, dropout=0., **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = K.floatx()
        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight((self.input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        return K.dot(x, self.W)


class Average(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            x = x * mask
        return K.sum(x, axis=-2) / K.sum(mask, axis=-2)

    def get_output_shape_for(self, input_shape):
        return input_shape[0:-2]+input_shape[-1:]
    
    def compute_mask(self, x, mask=None):
        return None


class MaxMargin(Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0] 
        z_n = input_tensor[1]
        r_s = input_tensor[2]

        z_s = z_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_s), axis=-1, keepdims=True)), K.floatx())
        z_n = z_n / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(z_n), axis=-1, keepdims=True)), K.floatx())
        r_s = r_s / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(r_s), axis=-1, keepdims=True)), K.floatx())

        steps = z_n.shape[1]

        pos = K.sum(z_s*r_s, axis=-1, keepdims=True)
        pos = K.repeat_elements(pos, steps, axis=-1)
        r_s = K.expand_dims(r_s, dim=-2)
        r_s = K.repeat_elements(r_s, steps, axis=1)
        neg = K.sum(z_n*r_s, axis=-1)

        loss = K.cast(K.sum(T.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), K.floatx())
        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)




        

from tensorflow.python.keras.layers import Layer, Add, BatchNormalization, Conv2D, SeparableConv2D, Conv2DTranspose, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.applications.vgg19 import VGG19
import tensorflow as tf

from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

LR_SIZE = 24
HR_SIZE = 96


class TransposeConv(Layer):
    def __init__(self, filter_size, num_filters, name, batch_size=1, scale=2):
        super(TransposeConv, self).__init__()
        self.filter_size = filter_size   # scaling by 2 for each upsample
        self.num_filters = num_filters   # number of output channels
        self.batch_size = batch_size     # declare input batch size explicitly for fixed graph
        self.scale = scale               
        self.stride = scale              # stride for 2x upscaling 
        self.layer_name = name           


    def build(self, input_shape):
        shape = [self.filter_size, self.filter_size,           # filter shape (only square)
               self.num_filters, input_shape[-1]]         
        output_size_h = (input_shape[1]- 1)*self.stride + self.filter_size  # output height with 'VALID' padding
        output_size_w = (input_shape[2]- 1)*self.stride + self.filter_size  # output with with 'VALID' padding
        self.out_shape = tf.stack([self.batch_size,                      # output tensor shape
                                output_size_h, output_size_w,
                                self.num_filters])
        self.filters = self.add_weight(                        # filter weights 
            name = self.layer_name+'w',
            shape=shape,
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(name = self.layer_name+'b',   # bias
            shape=(self.num_filters,), initializer=Constant, trainable=True
        )    

    def call(self, inputs):
        transconv = tf.nn.conv2d_transpose(inputs, self.filters, self.out_shape, self.stride, padding='VALID', data_format='NHWC', name=self.layer_name)
        return tf.nn.bias_add(transconv, self.b, data_format='NHWC')  # add bias

def upsample(x_in, num_filters):
    x = SeparableConv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = SeparableConv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = SeparableConv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(input_size=None, batch_size=1, num_filters=64, num_res_blocks=16, scale=4):
    if type(input_size)==tuple:
      shape = (input_size[0], input_size[1], 3)
    else:
      shape = (input_size, input_size, 3)
    x_in = Input(shape=shape)
    x = Lambda(lambda x: x)(x_in)
    x = SeparableConv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = SeparableConv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    # upscale by by 2x at a time
    x = TransposeConv(2, num_filters, 'uc1', batch_size=batch_size, scale=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    
    if scale>2:
        x = TransposeConv(2, num_filters, 'uc2', batch_size=batch_size, scale=2)(x)
        x = PReLU(shared_axes=[1, 2])(x)
    if scale>4:
        x = TransposeConv(2, num_filters, 'uc3', batch_size=batch_size, scale=2)(x)
        x = PReLU(shared_axes=[1, 2])(x)

    x = SeparableConv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    return Model(x_in, x)


generator = sr_resnet


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)

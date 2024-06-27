import tensorflow as tf
import numpy as np

RESIZE_FACTOR = 2

def resize_bilinear(x):
  return tf.compat.v1.image.resize_bilinear(x, size=[tf.shape(x)[1] * RESIZE_FACTOR, tf.shape(x)[2] * RESIZE_FACTOR])

def resize_output_shape(input_shape):
  shape = list(input_shape)
  assert len(shape) == 4
  shape[1] *= RESIZE_FACTOR
  shape[2] *= RESIZE_FACTOR
  return tuple(shape)

class EAST_model(tf.keras.Model):
  def __init__(self, input_size=512):
    super(EAST_model, self).__init__()

    input_image = tf.keras.layers.Input(shape=(None, None, 3), name='input_image')
    overly_small_text_region_training_mask = tf.keras.layers.Input(shape=(None, None, 1), name='overly_small_text_region_training_mask')
    text_region_boundary_training_mask = tf.keras.layers.Input(shape=(None, None, 1), name='text_region_boundary_training_mask')
    target_score_map = tf.keras.layers.Input(shape=(None, None, 1), name='target_score_map')
    resnet = tf.keras.applications.ResNet50(input_tensor=input_image, weights='imagenet', include_top=False, pooling=None)
    x = resnet.get_layer('conv5_block3_out').output
    x = self.fem(384, x)

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_1')(x)
    y = resnet.get_layer('conv4_block6_out').output
    y = self.fem(256,y);
    x = tf.keras.layers.concatenate([x, y], axis=3)
    #x = tf.keras.layers.concatenate([x, resnet.get_layer('conv4_block6_out').output], axis=3)
    x = tf.keras.layers.Conv2D(128, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_2')(x)
    y1 = resnet.get_layer('conv3_block4_out').output
    y1 = self.fem(128,y1)
    x = tf.keras.layers.concatenate([x, y1], axis=3)
    x = tf.keras.layers.Conv2D(64, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Lambda(resize_bilinear, name='resize_3')(x)
    y2 = resnet.get_layer('conv2_block3_out').output
    y2 = self.fem(64,y2)
    x = tf.keras.layers.concatenate([x, y2], axis=3)
    #x = tf.keras.layers.concatenate([x, resnet.get_layer('conv2_block3_out').output], axis=3)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)

    pred_score_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='pred_score_map')(x)
    rbox_geo_map = tf.keras.layers.Conv2D(4, (1, 1), activation=tf.nn.sigmoid, name='rbox_geo_map')(x)
    rbox_geo_map = tf.keras.layers.Lambda(lambda x: x * input_size)(rbox_geo_map)
    angle_map = tf.keras.layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, name='rbox_angle_map')(x)
    angle_map = tf.keras.layers.Lambda(lambda x: (x - 0.5) * np.pi / 2)(angle_map)
    pred_geo_map = tf.keras.layers.concatenate([rbox_geo_map, angle_map], axis=3, name='pred_geo_map')

    model = tf.keras.models.Model(inputs=[input_image], outputs=[pred_score_map, pred_geo_map])

    self.model = model
    self.input_image = input_image
    self.overly_small_text_region_training_mask = overly_small_text_region_training_mask
    self.text_region_boundary_training_mask = text_region_boundary_training_mask
    self.target_score_map = target_score_map
    self.pred_score_map = pred_score_map
    self.pred_geo_map = pred_geo_map

  def call(self, x):
    return self.model(x)

  def fem(self, no_of_filters, x):
    y1 = tf.keras.layers.Conv2D(no_of_filters, (3, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    y1 = tf.keras.layers.Conv2D(no_of_filters, (1, 3),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y1)
    y1 = tf.keras.layers.Conv2D(no_of_filters, (1, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y1)

    y2 = tf.keras.layers.Conv2D(no_of_filters, (5, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    y2 = tf.keras.layers.Conv2D(no_of_filters, (1, 5),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y2)
    y2 = tf.keras.layers.Conv2D(no_of_filters, (1, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y2)

    y3 = tf.keras.layers.Conv2D(no_of_filters, (7, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    y3 = tf.keras.layers.Conv2D(no_of_filters, (1, 7),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y3)
    y3 = tf.keras.layers.Conv2D(no_of_filters, (1, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y3)

    x = tf.keras.layers.Conv2D(no_of_filters, (1, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(y3)
    x = tf.keras.layers.concatenate([x, y1, y2, y3])
    x = tf.keras.layers.Conv2D(no_of_filters, (1, 1),  padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.997, epsilon=1e-5, scale=True)(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

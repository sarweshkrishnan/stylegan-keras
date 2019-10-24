import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from tensorflow.keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from tensorflow.keras.models import Model,model_from_json
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

import tensorflow_datasets as tfds
import tensorflow_gan as tfgan

from stylegan.generator import _generator,_style_mapping,AdaInstanceNormalization
from stylegan.discriminator import _discriminator
from stylegan.gan_model import GAN 

from functools import partial

import numpy as np
import json
from PIL import Image
import sys

def _predict(gan, show_fn=None):
	onesEval = np.ones((1,1),dtype=np.float32)

	n1 = np.random.normal(size = [1, gan.latent_size])
	latents = []
	for i in range(gan.style_layers):
		latents.append(gan.styler().predict(n1))

	images = gan.gen().predict(latents+[np.random.uniform(size = [1, gan.img_size, gan.img_size, 1]),onesEval])
	image_grid = tfgan.eval.python_image_grid(images, grid_shape=(1,1))
	if show_fn!=None:
		show_fn(image_grid)
	gan.saveGenerated(image_grid)

def create_input_model(gan):
  gen = gan.gen()
  gen.trainable = False
  for layer in gen.layers:
    layer.trainable = False

  input_latent = Input(shape=[gan.latent_size])
  x = Dense(gan.latent_size)(input_latent)
  x = Dense(gan.latent_size)(x)
  x = Dense(gan.latent_size)(x)
  x = Dense(gan.latent_size)(x)

  input_model = Model(input_latent, x)
  
  const_inp = Input(shape = [gan.img_size, gan.img_size, 1])
  const_1_inp = Input(shape = [1])
  x = gen([input_model.output] * gan.style_layers + [const_inp, const_1_inp])

  return Model(inputs=[input_latent, const_inp, const_1_inp], outputs=x)

def train(gan, model, true_latent):
  true_latents = []
  for i in range(gan.style_layers):
    true_latents.append(true_latent)
  true_img = gan.gen().predict(true_latents+[np.zeros((1,gan.img_size,gan.img_size,1)), np.ones((1,1),dtype=np.float32)])

  image_grid = tfgan.eval.python_image_grid(true_img, grid_shape=(1,1))
  saveGenerated("orig", image_grid)

  model.fit([np.ones((1, gan.latent_size)), np.zeros((1, gan.img_size, gan.img_size, 1)), np.ones((1, 1), dtype=np.float32)], [true_img], batch_size=1, steps_per_epoch=1, epochs=10000)

def main():
  np.random.seed(100) 
  gan = GAN(img_size=256, steps=1077000, preTrained=True)
 
  model = create_input_model(gan)
  model.summary()

  model.compile(optimizer = Adam(0.001, beta_1 = 0.9, beta_2 = 0.999), loss = 'mean_squared_error', metrics=['mse'])

  # get true latent/image
  n1 = np.random.normal(size = [1, gan.latent_size])
  true_latent = gan.styler().predict(n1)

  train(gan, model, true_latent)

  intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)
  predicted_latent = intermediate_layer_model.predict([np.ones((1, gan.latent_size)), np.random.uniform(size = [1, gan.img_size, gan.img_size, 1]), np.ones((1, 1), dtype=np.float32)])

  final_img = model.predict([np.ones((1, gan.latent_size)), np.zeros((1, gan.img_size, gan.img_size, 1)), np.ones((1, 1), dtype=np.float32)])
  image_grid = tfgan.eval.python_image_grid(final_img, grid_shape=(1,1))
  saveGenerated("final", image_grid)

  np.set_printoptions(threshold=sys.maxsize)
  print("True latent", true_latent)
  print("Predicted latent", predicted_latent)


def saveGenerated(name, img):
  img = Image.fromarray(np.uint8(img*255),mode = 'RGB')
  img.save("{0}.jpg".format(name))


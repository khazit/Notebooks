"""
TF2 implementation of some well established CNN architectures

		- VGG-16
"""


import tensorflow as tf
from tensorflow.keras import layers


class VGG(tf.keras.Model):

	def __init__(self, n_classes):
		super(VGG, self).__init__(name="vgg")
		self.n_classes = n_classes
		
		# 1st block
		self.conv1_1 = layers.Conv2D(
				filters=64, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.conv1_2 = layers.Conv2D(
				filters=64, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			) 
		self.pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

		# 2nd block
		self.conv2_1 = layers.Conv2D(
				filters=128, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.conv2_2 = layers.Conv2D(
				filters=128, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.pool2 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

		# 3rd block
		self.conv3_1 = layers.Conv2D(
				filters=256, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.conv3_2 = layers.Conv2D(
				filters=256, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)		
		self.conv3_3 = layers.Conv2D(
				filters=256, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)	 
		self.pool3 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

		# 4th block
		self.conv4_1 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.conv4_2 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)		
		self.conv4_3 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.pool4 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

		# 5th block
		self.conv5_1 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.conv5_2 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)		
		self.conv5_3 = layers.Conv2D(
				filters=512, 
				kernel_size=(3, 3), 
				padding="same",
				activation="relu"
			)
		self.pool5 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

		# Dense layers
		self.flat = layers.Flatten()
		self.dense1 = layers.Dense(units=4096, activation="relu")
		self.dropout1 = layers.Dropout(rate=0.5)
		self.dense2 = layers.Dense(units=4096, activation="relu")
		self.dropout2 = layers.Dropout(rate=0.5)
		self.dense3 = layers.Dense(n_classes)

	def call(self, inputs):
		# 1st block
		x = self.conv1_1(inputs)
		x = self.conv1_2(x) 
		x = self.pool1(x)

		# 2nd bock
		x = self.conv2_1(x)
		x = self.conv2_2(x) 
		x = self.pool2(x)

		# 3rd block
		x = self.conv3_1(x)
		x = self.conv3_2(x) 
		x = self.conv3_3(x) 
		x = self.pool3(x)

		# 4th block
		x = self.conv4_1(x)
		x = self.conv4_2(x) 
		x = self.conv4_3(x) 
		x = self.pool4(x)

		# 5th block
		x = self.conv5_1(x)
		x = self.conv5_2(x) 
		x = self.conv5_3(x) 
		x = self.pool5(x)
	 
		# Dense layers
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dropout1(x)
		x = self.dense2(x)
		x = self.dropout2(x)
		x = self.dense3(x)

		return x

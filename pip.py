import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_names = ['T-shirt/top' , 'Trouser' , 'Pullover' , 'Dress' , 'Coat' , 'Sandal' , 'Shirt' , 'Sneaker' , 'Bag' , 'Ankle_Boot']
train_images = train_images/255.0
test_images = test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc:", test_acc)
prediction = model.predict(test_images)
for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel("Actual:" + class_names[test_labels[i]])
	plt.title("Prediction :" + class_names[np.argmax(prediction[i])])
	plt.show()
mnist = tf.keras.datasets.mnist
classnames = ["1" , "2" , "3" , "4" , "5", "6", "7", " 8", "9"]
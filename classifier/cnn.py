import mnist_reader
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# data splits via mnist_reader functions
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind = 'train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind = 't10k')

# normalization
X_train = X_train / 255.0 
X_test = X_test / 255.0

# reshaping for CNN input
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# CNN object
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(28, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(56, (3, 3), activation = 'relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(56, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax'),
])
cnn.summary()

# validation samples
X_val = X_train[-12000:]
y_val = y_train[-12000:]
X_train = X_train[:-12000]
y_train = y_train[:-12000]

# Adam optimzer, crossentropy loss (sparase), batch size 32, 10 epoch training
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = cnn.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data = (X_val, y_val))

# training vs validation accuracy across epochs
plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc = 'lower right')

# evaluating CNN on test set
test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
print(test_accuracy)


# ALL SUBSEQUENT CODE IS FOR IDENTIFYING AND DISPLAYING MISCLASSIFCATION CASES TO GET A BETTER UNDERSTANDING OF THE MODEL
# label dict
labels = {0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

# predicting and encoding labels
y_pred = cnn.predict(X_test, verbose = 0)
y_pred_classes = np.argmax(y_pred, axis = 1)

# function for displaying incorrect instances
def displayMisclassifiedImage(label):
    for j in range(9999):            
        if y_test[j] == label:
                
            if y_pred_classes[j] != y_test[j]:
                plt.imshow(X_test[j], cmap = 'gray')
                # formatting title to include predicted label vs actual label
                plt.title("Predicted: {}\nTrue: {}".format(labels[y_pred_classes[j]], labels[y_test[j]]))  
                # break out of the loop once a single misclassified image is found
                break
              
# display misclasified image for each image type in fashion-mnist dataset
fig = plt.figure(figsize = (12, 6))
for i in range(10):
    fig.add_subplot(2, 5, i+1)
    displayMisclassifiedImage(i)


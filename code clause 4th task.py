import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic medical images
def generate_synthetic_images(num_images, image_shape):
    images = np.zeros((num_images, *image_shape))
    labels = np.random.randint(0, 2, num_images)  # Binary labels for demonstration
    for i in range(num_images):
        if labels[i] == 0:
            images[i] = np.random.normal(loc=0.2, scale=0.1, size=image_shape)  # Example of normal tissue
        else:
            images[i] = np.random.normal(loc=0.8, scale=0.1, size=image_shape)  # Example of abnormal tissue
    return images, labels

# Define your CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Generate synthetic data
num_images = 1000
image_shape = (128, 128, 3)  # Example image shape (height, width, channels)
X_synthetic, y_synthetic = generate_synthetic_images(num_images, image_shape)

# Define input shape and number of classes
input_shape = image_shape
num_classes = 2  # Binary classification for demonstration

# Create and compile the model
model = create_model(input_shape, num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with synthetic data
model.fit(X_synthetic, y_synthetic, epochs=10, validation_split=0.2)

# Evaluate the model with synthetic data
test_loss, test_acc = model.evaluate(X_synthetic, y_synthetic)
print('Test accuracy:', test_acc)

# Save the model
model.save('medical_image_classification_model.keras')

# Display example images
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(X_synthetic[i], cmap='gray')  # Display grayscale image
    plt.title('Label: {}'.format(y_synthetic[i]))
    plt.axis('off')
plt.show()


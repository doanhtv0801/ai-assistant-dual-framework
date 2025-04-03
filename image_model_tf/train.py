
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

x_train = np.random.rand(300, 64, 64, 3)
y_train = np.random.randint(0, 3, 300)

model = models.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=16)
model.save("image_model.h5")
print("✅ Image model trained and saved!")

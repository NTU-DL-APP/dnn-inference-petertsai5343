import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 下載並準備資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize & reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# 建模型 (簡單2層Dense)
model = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 評估模型
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# 儲存模型
model.save('fashion_mnist.h5')
print("Model saved as fashion_mnist.h5")

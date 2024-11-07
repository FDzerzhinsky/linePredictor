from keras_visualizer import visualizer
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(16, input_dim=2, activation='tanh')) # Входной слой с 4 нейронами
model.add(Dense(4, activation='tanh')) # Скрытый слой с 8 нейронами
model.add(Dense(1, activation='tanh')) # Выходной слой с 1 нейроном
model.save('model.h5')

visualizer(model, file_format='png')

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 1])

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Обучаем модель
model.fit(x_train, y_train, epochs=5, verbose=1)

# Оцениваем модель
loss, accuracy = model.evaluate(x_train, y_train, verbose=0)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Делаем прогнозы
predictions = model.predict(np.array([[1, 1]]))
print('Predictions:', predictions)
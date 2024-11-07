import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from keras.utils.vis_utils import plot_model



# Создаем синтетические данные
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Создаем модель персептрона
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# Компилируем модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучаем модель
model.fit(X, y, epochs=100, verbose=0)

# Оцениваем модель
loss, accuracy = model.evaluate(X, y, verbose=0)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Делаем прогнозы
predictions = model.predict(X)
print('Predictions:', predictions)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
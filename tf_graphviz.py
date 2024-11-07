import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Создаем модель нейронной сети
model = Sequential()
model.add(Dense(16, input_dim=4, activation='relu')) # Входной слой с 4 нейронами
model.add(Dense(8, activation='relu')) # Скрытый слой с 8 нейронами
model.add(Dense(1, activation='sigmoid')) # Выходной слой с 1 нейроном
model.save('model.h5')
# Визуализация модели с помощью TF-GraphViz
tf.keras.utils.plot_model(
  model,
  to_file='model_plot.png',
  show_shapes=True,
  show_layer_names=True,
  rankdir='LR', # Расположение слоев слева направо
  expand_nested=True, # Развернуть вложенные слои
)
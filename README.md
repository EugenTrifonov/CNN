# 1.Обучение нейронной сети,представленной в примере, для решения задачи классификации изображений Oregon Wildlife
## 1)Структура
Свёрточный слой с 8-ю фильтрами и размером ядра 3х3
```python
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
```
Слой MaxPool2D позволяет уменьшить дискретизацию данных посредством выбора максимального значения в окне 
```python
x = tf.keras.layers.MaxPool2D()(x)
```
Flatten приводит матрицу признаков к одномерному вектору 
```python
x = tf.keras.layers.Flatten()(x)
```
Полностью связанный слой с 20-ю выходами(NUM_CLASSES=20) и функцией активации softmax, которая приводит вероятностную оценку
```python
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## 2)Графики 
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr1](https://github.com/EugenTrifonov/CNN/blob/main/graphs/epoch_categorical_accuracy_1.svg)


Функция потерь
![gr2](https://github.com/EugenTrifonov/CNN/blob/main/graphs/epoch_loss_1.svg)
# 2.Создание и обучение сверточной нейронной сети произвольной архитектуры с количеством сверточных слоев >3
## 1)Структура
Были добавлены 3 свёрточных слоя с 8-ю фильтрами и размером ядра 3х3
```python
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
```
## 2)Графики
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![gr3](https://github.com/EugenTrifonov/CNN/blob/main/graphs/epoch_categorical_accuracy_2.svg)

Функция потерь

![gr4](https://github.com/EugenTrifonov/CNN/blob/main/graphs/epoch_loss_2.svg)

# 3.Анализ результатов
К исходной структере были добавлены 3 свёрточных слоя с 8-ю фильтрами и размером ядра 3х3.К улушчению качества обучения это не привело. Это можно отследить по графикам метрики качества и функции потерь. Добавление слоёв привело к увеличению глубины нейронной сети и, соответственно, процесс обучения стал более долгим.


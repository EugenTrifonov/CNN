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
Полностью связанный слой с 20-ю выходами и функцией активации softmax, которая приводит вероятностную оценку
```python
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
## 2)Графики 
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![epoch_categorical_accuracy (1)](https://user-images.githubusercontent.com/80068414/110242769-92bb3200-7f68-11eb-8d6a-2b2954b68ac8.png)


Функция потерь

![image](https://user-images.githubusercontent.com/80068414/110239506-3e0fbb00-7f58-11eb-9f3b-52c692fc9d57.png)
## 3)Анализ результатов
# 2.Создание и обучение сверточной нейронной сети произвольной архитектуры с количеством сверточных слоев >3
## 1)Структура
4 свёрточных слоя с 8-ю фильтрами и размером ядра 3х3
```python
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
```
## 2)Графики
![legend](https://user-images.githubusercontent.com/80068414/110239448-f25d1180-7f57-11eb-89d3-f19ba3d1d67a.png)

Метрика качества

![image](https://user-images.githubusercontent.com/80068414/110240428-eb84cd80-7f5c-11eb-8f0a-38cdeb58c2ca.png)

Функция потерь

![image](https://user-images.githubusercontent.com/80068414/110240433-f17aae80-7f5c-11eb-9345-ee56e8ea6e41.png)

## 3)Анализ результатов



import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler


def load_data(path='/content/microarray-data/data/', shuffle=False):
  """
  Funkcja działa analogicznie do load_data zbioru MNIST.
  Zwraca (X_train, y_train), (X_test, y_test) o wymiarach:
  (900, 32, 32)
  (900,)
  (157, 32, 32)
  (157,)

  Dla potrzeb wizualizacji każdy gen przyjmuje wymiary kwadratu 32 x 32.
  """
  microarrays = []

  # Załadowanie danych
  for filename in os.listdir(path):
    full_path = os.path.join(path, filename)

    if os.path.isfile(full_path):
      if filename[-5:] == '.arff':
        microarray, _ = arff.loadarff(full_path)
        microarrays.append(microarray)

  # Konkatenacja danych z plików do jednej listy o długości 1057
  data = [*microarrays[0], *microarrays[1], *microarrays[2], *microarrays[3],
          *microarrays[4], *microarrays[5], *microarrays[6], *microarrays[7]]

  # 1057 pacjentów
  patients = len(data)
  # 1000 genów i 1 etykieta na pacjenta
  genes = len(data[0]) - 1

  # Zamiana listy data na macierze numpy
  X_flatten = np.zeros((patients, genes))
  X_reshaped = np.zeros((patients, 32, 32))
  y = np.zeros(patients)

  for i in range(patients):
    for j in range(genes):
      X_flatten[i][j] = data[i][j]

    # Normalizacja w przedziale <0, 1> w obrębie pojedynczego genu
    df = pd.DataFrame(X_flatten[i])
    x = df.select_dtypes(include=[np.number])
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)

    # Dostawienie zer, aby długość genomu była równa 1024
    # I zmiana wymiaru x z (1000, 1) na 1000
    x = np.concatenate((np.reshape(x, 1000), np.zeros(24)))

    # Zmiana kształtu genu na 32 x 32
    X_reshaped[i] = np.reshape(x, (32, 32))

    # Zastąpienie etykiet wartościami logicznymi
    if str(data[i][1000]) == "b'yes'":
      y[i] = 1
    elif str(data[i][1000]) == "b'no'":
      y[i] = 0

  # Przetasowanie elementów
  if shuffle:
    indices = np.arange(patients)
    X = tf.gather(X_reshaped, indices)
    y = tf.gather(y, indices)
  else:
    X = X_reshaped

  # Podział na zbiór treningowy i testowy
  X_train = X[:900]
  X_test = X[-157:]
  y_train = y[:900]
  y_test = y[-157:]

  return (X_train, y_train), (X_test, y_test)


# Do użycia podczas trenowania GANów i sieci niewymagających etykiet
def load_data_by_label(label, path='/content/microarray-data/data/', shuffle=False):
  """
  Zwraca (X_train, X_test) o wymiarach:
  (n, 32, 32)
  (n, 32, 32)
  Gdzie n jest liczbą przypadków oznaczonych daną etykietą.
  """
  (X_train, y_train), (X_test, y_test) = load_data(path, shuffle=shuffle)

  assert X_train.shape[0] == y_train.shape[0]
  assert X_test.shape[0] == y_test.shape[0]
  assert label == 0 or label == 1

  X_pos_train = []
  for i in range(X_train.shape[0]):
    if y_train[i] == label:
      X_pos_train.append(X_train[i])

  X_pos_test = []
  for i in range(X_test.shape[0]):
    if y_test[i] == label:
      X_pos_test.append(X_test[i])

  X_pos_train = np.array(X_pos_train)
  X_pos_test = np.array(X_pos_test)

  return X_pos_train, X_pos_test


def convert_to_arff(images, labels, filename_r, filename_w, precision=3):
  """
  Konwersja obrazów z powrotem na plik .arff
  Funkcja przyjmuje argumenty:
  images o wymiarach (n, 32, 32)
  labels o wymiarach (n,)
  filename_r - nazwa pliku .arff, z którego pochodzą dane obrazów i etykiet
  filename_w - nazwa pliku .arff, do którego zostaną zapisane dane z obrazów np. po augmentacji
  precision - liczba cyfr po przecinku w danych, domyślnie 3

  n jest liczbą pacjentów (wierszy) we wczytanym pliku .arff !!
  """
  assert images.shape[1] == images.shape[2] == 32
  assert images.shape[0] == labels.shape[0]

  data_length = images.shape[0]
  genes = 1000

  file_r = open(filename_r, 'r')
  file_w = open(filename_w, 'w')

  # Przepisanie pierwszej linijki do nowego pliku
  lines = file_r.readlines()
  file_w.write(lines[0] + '\n')

  # Przepisanie artybutów
  for line in lines:
    if '@attribute' in line:
      file_w.write(line)

  file_w.write('\n@data \n')

  for i in range(data_length):
    img = np.array(images[i]).reshape(1024)[:1000]

    for j in range(genes):
      file_w.write(str(round(img[j], precision)) + ',')

    if labels[i] == 1:
      file_w.write("'yes' \n")
    elif labels[i] == 0:
      file_w.write("'no' \n")

  file_r.close()
  file_w.close()
  return file_w

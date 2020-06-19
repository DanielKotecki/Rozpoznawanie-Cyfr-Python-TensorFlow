from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

width = 200
height = 200
center = height // 2
white = (255, 255, 255)
green = (0, 128, 0)


def funkcjaMyslaca():

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=8)
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)
    model.save('pismo_reczne.model')
    img = cv2.imread('image.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"Prawdopodobnie jest to: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()


def sprawdz():
    sciezka = "image.png"
    nowyObrazek = image1.resize((28, 28))
    nowyObrazek.save(sciezka)
    funkcjaMyslaca()


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=6)
    draw.line([x1, y1, x2, y2], fill="black", width=6)


okno = Tk()
okno.title("Program do rozpoznawania liczb od 0-9")
cv = Canvas(okno, width=width, height=height, bg='white')
cv.pack()
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
button = Button(text="Sprawdz jaka to liczba", command=sprawdz)
button.pack()
okno.mainloop()

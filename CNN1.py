#IMAGE RECOGNITION CNN MODEL
import os
import tensorflow as tf 
import numpy as np

import cv2 

model = tf.keras.models.load_model('my_cnn_model.keras')   # LOADING CUSTOM TRAINED MODEL

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
def predict(path):
    img = cv2.imread(path)
    if img is None:
        print("Failed To Open Image")
        return
    img = cv2.resize(img,(32,32))
    img = img/255.0000
    img = np.expand_dims(img,axis=0)
    preds = model.predict(img)
    pred_class = classes[np.argmax(preds[0])]
    print("This is a : ",pred_class)

while True:
    path = input("Enter Picture Path : ('q' to stop )")
    if path.lower() == 'q':
        break
    predict(path)





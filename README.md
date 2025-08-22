CNN Image Classifier (CIFAR-10)
A simple, end-to-end Convolutional Neural Network project for image classification on the CIFAR-10 dataset. Trains a Keras/TensorFlow model with data augmentation, evaluates on test data, saves the model, and includes a small CLI to run predictions on custom images using OpenCV.

Features
CNN built with TensorFlow/Keras

Data augmentation via ImageDataGenerator (rotation, shift, flip)

Trains with generator; validates on held-out test set

Accuracy/loss plots with Matplotlib

Model saving (.keras) and zipped download (Colab-friendly)

CLI image prediction using OpenCV + NumPy

Class labels: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Tech Stack
Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Google Colab (optional)

Project Structure
CNN1.py: Training script (augmentation, training loop, saving, plotting, download zip)

predict.py: Load saved model and run CLI predictions on images

my_cnn_model.keras: Saved Keras model (generated after training)

my_cnn_model.zip: Archived model for download (generated)

sample images: img1.png, img2.png, img3.png, img5.png (for quick testing)

Note: Filenames can vary based on the environment.

Dataset
CIFAR-10 (10 classes, 60,000 images, 32x32). In Keras: tf.keras.datasets.cifar10.load_data()

Setup
Create environment

Python 3.9+ recommended

Install dependencies
pip install tensorflow numpy opencv-python matplotlib

(Optional) Run in Google Colab

This repo works out of the box in Colab.

The training script includes archive/download steps for the model:

shutil.make_archive('my_cnn_model', 'zip', '.', 'my_cnn_model.keras')

files.download('my_cnn_model.zip')

Training
Run the training script:
python CNN1.py

What it does:

Loads CIFAR-10

Normalizes images to

Applies augmentation:

rotation_range=15

width_shift_range=0.1

height_shift_range=0.1

horizontal_flip=True

Trains with:

optimizer='adam'

loss='sparse_categorical_crossentropy'

metrics=['accuracy']

epochs=10 (adjust as needed)

batch_size=64

Evaluates on test set and prints test accuracy

Saves the model to my_cnn_model.keras

Archives and (in Colab) downloads my_cnn_model.zip

Plots training/validation accuracy

Adjust epochs/batch size in the script for better results or faster runs.

Inference (CLI Prediction)
After training (or downloading the provided model), use predict.py:

python predict.py

It will load my_cnn_model.keras

Enter an image path when prompted

Type q to exit

Example:
Enter Picture Path : ('q' to stop ) img1.png
This is a : cat

Tips:

Images are auto-resized to 32x32 and normalized.

Works best on images roughly matching CIFAR-10 categories.


CLI prediction (core steps):
img = cv2.imread(path)
img = cv2.resize(img, (32, 32))
img = img / 255.0
img = np.expand_dims(img, axis=0)
preds = model.predict(img)
pred_class = classes[np.argmax(preds)]

Customization
Change model architecture: edit the model definition in CNN1.py

Tune augmentation: modify ImageDataGenerator params

Change optimizer/loss: update model.compile

Increase epochs or tweak batch size for better accuracy

Replace CIFAR-10 with a custom dataset (ensure 32x32 or adjust input shape and classes)

Results
Prints final test accuracy in console

Shows training vs validation accuracy plot

Real-world predictions via predict.py

Note: Actual accuracy will vary based on model architecture and training duration. Try increasing epochs or adding regularization (Dropout, BatchNorm) for improvements.

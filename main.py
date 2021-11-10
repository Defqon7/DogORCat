import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import os
import random
from shutil import copyfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Dogs')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Cats')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Dogs')
    os.mkdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Cats')
except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " has a length of zero.")

        training_length = int(len(files) * SPLIT_SIZE)
        testing_length = int(len(files) - training_length)
        shuffled_set = random.sample(files, len(files))  # shuffle the files
        training_set = shuffled_set[0:training_length]  # from index 0 to training_length
        testing_set = shuffled_set[-testing_length:]  # rest of files

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/Cat/'
TRAINING_CATS_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Cats/'
TESTING_CATS_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Cats/'
DOG_SOURCE_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/Dog/'
TRAINING_DOGS_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Dogs/'
TESTING_DOGS_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Dogs/'

if not os.path.exists('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd'):
    split_size = .9
    split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
    split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Cats')))
print(len(os.listdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/Dogs')))
print(len(os.listdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Cats')))
print(len(os.listdir('C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/Dogs')))

# model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=.001),
              metrics=['acc'])

# pre process images
TRAINING_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Training/'
train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    target_size=(150, 150),
                                                    batch_size=250,
                                                    class_mode='binary')
VALIDATION_DIR = 'C:/Users/hanna/OneDrive/Desktop/Datasets/PetImages/cvd/Testing/'
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              target_size=(150, 150),
                                                              batch_size=250,
                                                              class_mode='binary')

history = model.fit(train_generator, epochs=20, steps_per_epoch=90,
                    validation_data=validation_generator, validation_steps=6)

# plot acc and loss

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()


plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()

# save model
model.save('models/cat_or_dog.h5')
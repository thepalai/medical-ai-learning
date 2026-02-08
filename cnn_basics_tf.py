import os
import tensorflow as tf

DATA_DIR = 'horse-or-human'
#____________3 Ways to split dataset to training and validation_________________#

#________________________First Manually____________________________#

HORSES_DIR = os.path.join(DATA_DIR, 'horses')
HUMANS_DIR = os.path.join(DATA_DIR, 'humans')

horses_path = [os.path.join(HORSES_DIR,f) for f in os.listdir(HORSES_DIR)]
humans_path = [os.path.join(HUMANS_DIR,f) for f in os.listdir(HUMANS_DIR)]

files = horses_path + humans_path
labels = [0]*len(horses_path)+[1]*len(humans_path)

dataset = tf.data.Dataset.from_tensor_slices((files, labels))
dataset = dataset.shuffle(buffer_size=len(files), seed=42)
dataset_size = len(files)
train_size = int(0.9 * dataset_size)   # 90% train
val_size = dataset_size - train_size
train_ds = dataset.take(train_size)
val_ds   = dataset.skip(train_size)


#________________Second from sklearn.model_selection______________________#
from sklearn.model_selection import train_test_split

HORSES_DIR = os.path.join(DATA_DIR, 'horses')
HUMANS_DIR = os.path.join(DATA_DIR, 'humans')

horses_path = [os.path.join(HORSES_DIR,f) for f in os.listdir(HORSES_DIR)]
humans_path = [os.path.join(HUMANS_DIR,f) for f in os.listdir(HUMANS_DIR)]

files = horses_path + humans_path
labels = [0]*len(horses_path)+[1]*len(humans_path)
train_files, val_files, train_labels, val_labels = train_test_split(
    files, labels,
    test_size=0.1,     # 10 %
    stratify=labels,   # keeps class balance
    random_state=42
) # Shuffle default=True

train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
val_ds   = tf.data.Dataset.from_tensor_slices((val_files, val_labels))


#________________________Third Using Keras Function (Recommend)____________________________#

def split_train_val(dir):
    train, val = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dir,
        validation_split=0.1,
        subset="both",
        label_mode='binary',
        image_size=(150, 150),
        shuffle=False,
        batch_size=None,
        seed=42,
    )
    return train, val

train_ds, val_ds = split_train_val(DATA_DIR) # using only DATA_DIR

#_____________________________________________________________________________#


class EarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.98:
            print("\nAccuracy > 98% so stopping training")
            self.model.stop_training = True


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
])

    model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

model = create_model()



SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 32
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

train_dataset_final = train_ds.cache().shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(PREFETCH_BUFFER_SIZE)
validation_dataset_final = val_ds.cache().prefetch(PREFETCH_BUFFER_SIZE)
history = model.fit(train_dataset_final, validation_data=validation_dataset_final,epochs=10,callbacks=[EarlyStopping()])

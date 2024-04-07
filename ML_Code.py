#!/usr/bin/env python
# coding: utf-8

# <h1><b>LIBRARIES</h1>

# In[11]:


import os
import wave
#import librosa
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio


# <h1><b>VALUE ASSIGNMENT</h1>

# In[12]:


VALID_SPLIT = 0.1
SHUFFLE_SEED = 43
SAMPLING_RATE = 16000
BATCH_SIZE = 7
EPOCHS = 25


# <h1><b>CREATING AND STORING IN FOLDERS</h1>

# In[13]:


C_SUBF_N = "Clean"
N_SUBF_N = "Noise"

FOLD = os.path.join(os.path.expanduser("~"), r"C:\Users\Mohit Chand\Desktop\Git\SVR\finale\raw_audio")
C_SUBF = os.path.join(FOLD, C_SUBF_N)
N_SUBF = os.path.join(FOLD, N_SUBF_N)

print(FOLD)
print(C_SUBF)
print(N_SUBF)


if os.path.exists(C_SUBF) is False:
    os.makedirs(C_SUBF)


if os.path.exists(N_SUBF) is False:
    os.makedirs(N_SUBF)

for folder in os.listdir(FOLD):
    if os.path.isdir(os.path.join(FOLD, folder)):
        if folder in [C_SUBF_N, N_SUBF_N]:
            continue
        elif folder in ["other", "_background_noise_"]:
            shutil.move(
                os.path.join(FOLD, folder),
                os.path.join(N_SUBF, folder),
            )
        else:
            shutil.move(
                os.path.join(FOLD, folder),
                os.path.join(C_SUBF, folder),
            )


# <h1><b>DOWNSAMPLING FROM 24 BITS TO 16 BITS </h1>

# In[14]:


def convert_and_save_audio(root_folder):
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        for file in os.listdir(subfolder_path):
            if file.endswith('.wav'):
                audio_file_path = os.path.join(subfolder_path, file)
                with wave.open(audio_file_path, 'rb') as audio_file:
                    num_frames = audio_file.getnframes()
                    sample_width = audio_file.getsampwidth()
                    num_channels = audio_file.getnchannels()
                    frame_rate = audio_file.getframerate()
                    parameters = audio_file.getparams()

                    frames = audio_file.readframes(-1)

                    sample_size = num_frames * sample_width * num_channels

                    aud_new = wave.open(os.path.join(subfolder_path, file), "wb")
                    aud_new.setnchannels(2)
                    aud_new.setsampwidth(2)
                    aud_new.setframerate(16000.0)

                    aud_new.writeframes(frames)

                    aud_new.close()


if __name__ == "__main__":
    root_folder = r"C:\Users\Mohit Chand\Desktop\Git\SVR\finale\raw_audio\Clean"
    convert_and_save_audio(root_folder)
    


# <h1><b>FEATURE EXTRACTION</h1>

# In[15]:


def paths_and_labels_to_dataset(audio_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths) 
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
  audio = tf.io.read_file(path)
  audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
  return audio


def audio_to_fft(audio):
  audio = tf.squeeze(audio, axis=-1)
  fft = tf.signal.fft(
      tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
      )
  fft = tf.expand_dims(fft, axis=-1)
  return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])




class_names = os.listdir(C_SUBF)
print("Our class names: {}".format(class_names,))
audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print("Processing speaker {}".format(name,))
    dir_path = Path(C_SUBF) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)


# <h1><b>TRAINING</h1>

# In[16]:


rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(audio_paths)
rng = np.random.RandomState(SHUFFLE_SEED)
rng.shuffle(labels)


num_val_samples = int(VALID_SPLIT * len(audio_paths))
print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
train_audio_paths = audio_paths[:-num_val_samples]
train_labels = labels[:-num_val_samples]

print("Using {} files for validation.".format(num_val_samples))
valid_audio_paths = audio_paths[-num_val_samples:]
valid_labels = labels[-num_val_samples:]


train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
train_ds = train_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)


train_ds = train_ds.map(
      lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = valid_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)


# <h1><b>MODEL</h1>

# In[17]:


def residual_block(x, filters, conv_num=3, activation="relu"):
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape, name="input")

    x = residual_block(inputs, 16, 2)
    x = residual_block(x, 32, 2)
    x = residual_block(x, 64, 3)
    x = residual_block(x, 128, 3)
    #x = residual_block(x, 128, 3)

    x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)


model = build_model((SAMPLING_RATE // 2, 1), len(class_names))

model.summary()


model.compile(
    optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_save_filename = "SVR.h5"

earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)


# <h1><b>FIT TRAINING</h1>

# In[18]:


history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    callbacks=[earlystopping_cb, mdlcheckpoint_cb],
)


# <h1><b>CROSS VALIDATION</h1>

# In[19]:


print(model.evaluate(valid_ds))


# <h1><b>TESTING</h1>

# In[20]:


SAMPLES_TO_DISPLAY = 7

test_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)

for audios, labels in test_ds.take(1):
   
    ffts = audio_to_fft(audios)
    y_pred = model.predict(ffts)
    
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(SAMPLES_TO_DISPLAY):
       
        print(
            "Speaker:\33{} {}\33[0m\tPredicted:\33{} {}\33[0m".format(
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[labels[index]],
                "[92m" if labels[index] == y_pred[index] else "[91m",
                class_names[y_pred[index]],
            )
        )
        display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))


# In[ ]:





# In[ ]:





# In[ ]:





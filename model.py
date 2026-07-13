import os
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "dataset_2digit", "train")
TEST_PATH = os.path.join(BASE_DIR, "dataset_2digit", "test")

IMG_SIZE = 64

# =====================================
# LOAD TRAIN DATA
# =====================================

X_train = []
y_train = []

print("Loading Training Images...")

for label in range(100):

    folder = os.path.join(TRAIN_PATH, f"{label:02d}")

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype("float32") / 255.0

        X_train.append(img)
        y_train.append(label)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)

# =====================================
# LOAD TEST DATA
# =====================================

X_test = []
y_test = []

print("Loading Testing Images...")

for label in range(100):

    folder = os.path.join(TEST_PATH, f"{label:02d}")

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img.astype("float32") / 255.0

        X_test.append(img)
        y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = np.array(y_test)

# =====================================
# ONE HOT ENCODING
# =====================================

y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

print("Training Images :", len(X_train))
print("Testing Images  :", len(X_test))

# =====================================
# CNN MODEL
# =====================================

model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu",
                 input_shape=(64,64,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(100, activation="softmax"))

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# TRAIN
# =====================================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=20,

    batch_size=32,
    shuffle=True
)

# =====================================
# TEST
# =====================================

loss, accuracy = model.evaluate(X_test, y_test)

print("\nAccuracy :", accuracy)
# =====================================
# SHOW SAMPLE PREDICTIONS
# =====================================

# =====================================
# PREDICT ONE SAMPLE
# =====================================

index = 0    # Change this to test different images

prediction = model.predict(X_test[index:index+1], verbose=0)

predicted = np.argmax(prediction)
actual = np.argmax(y_test[index])

print("\nPrediction Result")
print("-----------------")
print("Predicted :", f"{predicted:02d}")
print("Actual    :", f"{actual:02d}")


# =====================================
# SAVE MODEL
# =====================================

model.save("digit_2digit_model.keras")

print("\nModel Saved Successfully!")
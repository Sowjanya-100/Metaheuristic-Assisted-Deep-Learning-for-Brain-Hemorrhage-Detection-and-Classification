import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# -------------------------
# 1️⃣ Paths
# -------------------------
train_dir = "preprocessed_dataset/train"
test_dir  = "preprocessed_dataset/test"

# -------------------------
# 2️⃣ Parameters
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

# -------------------------
# 3️⃣ Load Dataset
# -------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_ds = train_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# -------------------------
# 4️⃣ Data Augmentation
# -------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -------------------------
# 5️⃣ Transfer Learning (KEY)
# -------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -------------------------
# 6️⃣ Model
# -------------------------
model = models.Sequential([
    layers.Input(shape=(224,224,3)),
    layers.Rescaling(1./255),
    data_augmentation,

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

# -------------------------
# 7️⃣ Compile
# -------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# 8️⃣ Callbacks
# -------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# -------------------------
# 9️⃣ Train
# -------------------------
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -------------------------
# 🔟 Accuracy Curves
# -------------------------
epochs_range = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")
plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -------------------------
# 🔟 Loss Curves
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(epochs_range, history.history['loss'], label="Training Loss")
plt.plot(epochs_range, history.history['val_loss'], label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -------------------------
# 1️⃣1️⃣ Evaluate
# -------------------------
loss, acc = model.evaluate(test_ds)
print(f"Final Test Accuracy: {acc:.4f}")

# -------------------------
# 1️⃣2️⃣ Save
# -------------------------
model.save("brain_hemorrhage_cnn_model.h5")

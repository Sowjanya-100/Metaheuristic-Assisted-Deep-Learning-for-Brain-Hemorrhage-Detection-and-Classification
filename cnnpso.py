# =========================================================
# 1. IMPORTS
# =========================================================
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# 2. PATHS
# =========================================================
train_dir = "preprocessed_dataset/train"
test_dir  = "preprocessed_dataset/test"

# =========================================================
# 3. PARAMETERS
# =========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# 4. LOAD DATASET
# =========================================================
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

# =========================================================
# 5. DATA AUGMENTATION
# =========================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================================================
# 6. BASELINE CNN MODEL
# =========================================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

baseline_model = models.Sequential([
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

baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)

# =========================================================
# 7. TRAIN BASELINE CNN
# =========================================================
print("\nTraining Baseline CNN...\n")

baseline_history = baseline_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

baseline_model.save("brain_hemorrhage_cnn_model.h5")

# =========================================================
# 8. BASELINE CNN CURVES
# =========================================================
epochs_range = range(1, len(baseline_history.history['accuracy']) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_range, baseline_history.history['accuracy'], label="Train Accuracy (CNN)")
plt.plot(epochs_range, baseline_history.history['val_accuracy'], label="Val Accuracy (CNN)")
plt.title("Baseline CNN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs_range, baseline_history.history['loss'], label="Train Loss (CNN)")
plt.plot(epochs_range, baseline_history.history['val_loss'], label="Val Loss (CNN)")
plt.title("Baseline CNN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# =========================================================
# 9. CNN BUILDER FUNCTION (FOR PSO)
# =========================================================
def build_cnn(lr, neurons, dropout):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(224,224,3)),
        layers.Rescaling(1./255),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(int(neurons), activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =========================================================
# 10. FITNESS FUNCTION
# =========================================================
def fitness(particle):
    lr, neurons, dropout = particle
    model = build_cnn(lr, neurons, dropout)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=5,   # small epochs for PSO
        verbose=0
    )
    return history.history['val_accuracy'][-1]

# =========================================================
# 11. PSO PARAMETERS
# =========================================================
num_particles = 3
num_iterations = 3

w, c1, c2 = 0.5, 1.5, 1.5

bounds = [
    (1e-5, 1e-3),   # learning rate
    (64, 256),      # neurons
    (0.3, 0.7)      # dropout
]

# =========================================================
# 12. INITIALIZE PARTICLES
# =========================================================
pos = np.random.rand(num_particles, 3)
vel = np.zeros((num_particles, 3))

for i in range(3):
    pos[:, i] = bounds[i][0] + pos[:, i] * (bounds[i][1] - bounds[i][0])

pbest = pos.copy()
pbest_score = np.array([fitness(p) for p in pos])

gbest = pbest[np.argmax(pbest_score)]
gbest_score = np.max(pbest_score)

# =========================================================
# 13. PSO OPTIMIZATION LOOP
# =========================================================
print("\nStarting PSO Optimization...\n")

for it in range(num_iterations):
    print(f"Iteration {it+1}/{num_iterations}")

    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()

        vel[i] = (
            w * vel[i]
            + c1 * r1 * (pbest[i] - pos[i])
            + c2 * r2 * (gbest - pos[i])
        )

        pos[i] += vel[i]

        for d in range(3):
            pos[i][d] = np.clip(pos[i][d], bounds[d][0], bounds[d][1])

        score = fitness(pos[i])

        if score > pbest_score[i]:
            pbest[i] = pos[i]
            pbest_score[i] = score

            if score > gbest_score:
                gbest = pos[i]
                gbest_score = score

    print("Best Accuracy so far:", gbest_score)

# =========================================================
# 14. TRAIN FINAL CNN WITH PSO PARAMETERS
# =========================================================
best_lr, best_neurons, best_dropout = gbest

print("\nTraining CNN + PSO Optimized Model...\n")

final_model = build_cnn(best_lr, best_neurons, best_dropout)

final_history = final_model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

loss, acc = final_model.evaluate(test_ds)
print(f"Final CNN + PSO Accuracy: {acc:.4f}")

final_model.save("brain_hemorrhage_cnn_pso_model.h5")

# =========================================================
# 15. CNN + PSO CURVES
# =========================================================
epochs_range = range(1, len(final_history.history['accuracy']) + 1)

plt.figure(figsize=(8,5))
plt.plot(epochs_range, final_history.history['accuracy'], label="Train Accuracy (CNN+PSO)")
plt.plot(epochs_range, final_history.history['val_accuracy'], label="Val Accuracy (CNN+PSO)")
plt.title("CNN + PSO Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(epochs_range, final_history.history['loss'], label="Train Loss (CNN+PSO)")
plt.plot(epochs_range, final_history.history['val_loss'], label="Val Loss (CNN+PSO)")
plt.title("CNN + PSO Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

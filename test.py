import tensorflow as tf
import tensorflow_datasets as tfds

# Make sure TensorFlow version 2.x is being used.
# If not, the program will stop here.
assert tf.__version__.startswith("2.")

# Load the Iris dataset.
# split=['train[:80%]', 'train[80%:]'] means:
# - use the first 80% for training
# - use the last 20% for testing
#
# as_supervised=True means each item is returned as:
# (features, label)
#
# with_info=True gives extra information about the dataset.
(ds_train, ds_test), ds_info = tfds.load(
    "iris",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True
)

# This function changes each class label into one-hot form.
# Example:
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]
def preprocess(features, label):
    label = tf.one_hot(label, depth=3)
    return features, label

# Apply preprocessing to each example in the training set.
# batch(32) groups data into batches of 32 examples.
# prefetch(...) helps performance by preparing data in advance.
train_data = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Do the same for the test set.
test_data = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Build the neural network model.
# Input has 4 values because each iris flower has 4 measurements.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),                # Input layer: 4 features
    tf.keras.layers.Dense(128, activation="relu"),  # Hidden layer with 128 neurons
    tf.keras.layers.Dense(64, activation="relu"),   # Hidden layer with 64 neurons
    tf.keras.layers.Dense(3, activation="softmax")  # Output layer: 3 classes
])

# Compile the model.
# optimizer="adam" controls how learning happens.
# loss="categorical_crossentropy" is used for multi-class classification
# when labels are one-hot encoded.
# metrics=["accuracy"] tells TensorFlow to also report accuracy.
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model using the training data.
# epochs=50 means the model sees the training data 50 times.
# validation_data=test_data lets us check performance during training.
history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data
)

# Evaluate the trained model on the test data.
# This gives the final loss and accuracy.
loss, accuracy = model.evaluate(test_data)

# Print the test results.
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Create one new flower sample for prediction.
# These are the 4 input values:
# [sepal length, sepal width, petal length, petal width]
sample_data = tf.constant([[5.1, 3.3, 1.7, 0.5]])

# Ask the model to predict class probabilities for this sample.
predictions = model.predict(sample_data)

# Pick the class with the highest predicted probability.
predicted_class = tf.argmax(predictions, axis=1).numpy()

# Print the predicted class number.
print(f"Predicted class: {predicted_class}")
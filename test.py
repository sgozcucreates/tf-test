import tensorflow as tf
import tensorflow_datasets as tfds

# Ensure TensorFlow 2.x is used
assert tf.__version__.startswith('2.')

# Load the Iris dataset with defined splits for training and testing
(ds_train, ds_test), ds_info = tfds.load(
    'iris',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,  # Load the dataset as (feature, label) pairs
    with_info=True       # Retrieve the metadata of the dataset
)

# Function to preprocess the dataset by one-hot encoding the labels
def preprocess(features, label):
    label = tf.one_hot(label, depth=3)  # 3 classes in the Iris dataset
    return features, label

# Prepare the train and test datasets
train_data = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Define a simple neural network for classification
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),  # Input shape is 4 for the four Iris features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Softmax for multi-class classification
])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data and validate on the test data
history = model.fit(train_data, epochs=50, validation_data=test_data)

# Evaluate the model's performance on the test dataset
loss, accuracy = model.evaluate(test_data)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Use the model to make predictions on a new sample
sample_data = tf.constant([[5.1, 3.3, 1.7, 0.5]])  # Example measurements of an iris flower

# Make the predictions
predictions = model.predict(sample_data)
predicted_class = tf.argmax(predictions, axis=1).numpy()

print(f"Predicted class: {predicted_class}")
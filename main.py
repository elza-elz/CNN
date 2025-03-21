from keras.datasets import imdb
from keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Parameters
max_features = 5000
max_words = 500

# Load the IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
print(f'{len(X_train)} train sequences\n{len(X_test)} test sequences')

# Pad sequences to fixed length
X_train = pad_sequences(X_train, maxlen=max_words)
X_test = pad_sequences(X_test, maxlen=max_words)
print('train data shape: ', X_train.shape)
print('test data shape: ', X_test.shape)

# Build the model
model = models.Sequential()
model.add(layers.Embedding(max_features, 32, input_length=max_words))
model.add(layers.SimpleRNN(100))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test, y_test)

# Visualization of training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predict on a specific test example
test_seq = pad_sequences([X_test[7]], maxlen=max_words)
pred = model.predict(test_seq)[0]

# Check the predicted class
if pred[0] == 1:
    print('Positive Review')
else:
    print('Negative Review')

# Print the actual label
print('Actual label:', y_test[7])

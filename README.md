# Fashion MNIST Image Classification with Neural Networks

## 🧾 Overview

This project demonstrates a simple deep learning workflow using the **Fashion-MNIST** dataset. It uses **Keras** (TensorFlow backend) to build and train a **fully connected neural network** to classify grayscale images of clothing into one of 10 categories.

## 👚 Fashion-MNIST Dataset

- 70,000 images (60,000 training, 10,000 testing)
- 28x28 grayscale images
- 10 categories of fashion items:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## 🚀 Project Workflow

1. **Import Libraries**: TensorFlow/Keras, NumPy, Matplotlib
2. **Load Dataset**: Automatically from `keras.datasets.fashion_mnist`
3. **Data Preprocessing**:
   - Normalize image pixel values to [0, 1]
   - Reshape inputs for compatibility with the neural network
4. **Build Model**:
   - Input layer: 784 units (flattened 28x28 image)
   - Hidden layers: Dense layers with ReLU
   - Output layer: Softmax for 10-class classification
5. **Compile Model**:
   - Optimizer: Adam
   - Loss Function: Sparse Categorical Crossentropy
   - Metrics: Accuracy
6. **Train the Model**
7. **Evaluate Accuracy on Test Set**
8. **Predict and Visualize Results**

## 🧠 Model Summary (Example)

```python
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

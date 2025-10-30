# 🖐️ Sign Language MNIST ReadMe

A deep learning project that classifies **American Sign Language (ASL)** alphabets using the **Sign Language MNIST dataset**.  
The model is implemented using **TensorFlow** and **Keras**, and trained on grayscale 28×28 images representing 24 ASL letters (A–Y, excluding J and Z due to motion).

---

## 📘 Overview

This project demonstrates how to recognize **hand gestures of ASL alphabets** using a fully connected neural network (Multilayer Perceptron).  
It is designed to help bridge communication barriers for people with hearing or speech impairments through AI-driven gesture recognition.

---

## 📊 Dataset

The dataset used is the **Sign Language MNIST** dataset — a modified version of MNIST that contains labeled grayscale images of hand gestures.  
Each image is 28×28 pixels and represents a letter from A to Y (excluding J and Z).

Dataset

File

Samples

Description

Training

`sign_mnist_train.csv`

27,455

Grayscale 28×28 images of hand gestures

Testing

`sign_mnist_test.csv`

7,172

Evaluation set for trained model

📁 **Data files:**

-   `sign_mnist_train.csv`
-   `sign_mnist_test.csv`

---

## 🧠 Model Architecture

The model is a simple yet effective **fully connected neural network** built using the **Sequential API** from Keras.

```python
model = tf.keras.models.Sequential([    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),    tf.keras.layers.Dense(64, activation='relu'),    tf.keras.layers.Dense(24, activation='softmax')])
```

-   **Input layer:** Flattened 28×28 (784 features)
-   **Hidden layers:** Dense(128, ReLU) → Dense(64, ReLU)
-   **Output layer:** Dense(24, Softmax)
-   **Loss:** `sparse_categorical_crossentropy`
-   **Optimizer:** `Adam`
-   **Metrics:** Accuracy

---

## ⚙️ Installation

To run this project locally, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Temuulen-Munkhtaivan/Sign_Language_MNIST.gitcd Sign_Language_MNISTpip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 🚀 How to Run

1.  Launch Jupyter Notebook or VSCode.
    
2.  Open the file:
    
    ```bash
    SL_MNIST.ipynb
    ```
    
3.  Run all cells sequentially to:
    
    -   Load and preprocess dataset
    -   Train the neural network
    -   Evaluate and visualize performance

---

## 📈 Training & Results

-   **Training accuracy:** ~78%
-   **Testing accuracy:** ~75%
-   **Epochs:** 8
-   **Batch size:** 32

Sample training output:

```
Epoch 8/8 loss: 0.64 - accuracy: 0.7540
```

### Confusion Matrix and Accuracy Plot (example placeholders)

You can visualize accuracy and loss trends across epochs using:

```python
plt.plot(history.history['accuracy'], label='train_accuracy')plt.plot(history.history['val_accuracy'], label='val_accuracy')plt.legend()
```

---

## 🖼️ Visual Samples

### Example Images from the Dataset

Color Reference:  
![American Sign Language Reference](/archive/amer_sign2.png)

Grayscale Training Samples:  
![Training Samples](/archive/amer_sign3.png)

Illustrated Reference Chart:  
![ASL Alphabet Illustration](/archive/american_sign_language.PNG)

---

## 🧩 Future Improvements

-   Use **Convolutional Neural Networks (CNNs)** for spatial feature extraction
-   Implement **real-time sign detection** with a webcam (OpenCV + TensorFlow)
-   Expand dataset with **dynamic gestures (J, Z)**
-   Integrate a **web interface** using Flask or Streamlit

---

## 👨‍💻 Author

**Temuulen Munkhtaivan**  
🔗 [GitHub Profile](https://github.com/Temuulen-Munkhtaivan)

---

## 📚 References

-   [Sign Language MNIST Dataset (Kaggle)](https://www.kaggle.com/datamunge/sign-language-mnist)
-   TensorFlow Documentation: [https://www.tensorflow.org](https://www.tensorflow.org)
-   Keras API Reference: [https://keras.io](https://keras.io)

---

## 🧾 Citation

If you use this repository in your work, please cite it as:

```
@software{Temuulen_SL_MNIST,  author = {Munkhtaivan, Temuulen},  title = {Sign Language MNIST},  year = {2025},  url = {https://github.com/Temuulen-Munkhtaivan/Sign_Language_MNIST}}
```
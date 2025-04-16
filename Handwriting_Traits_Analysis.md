
# Personality Trait Detection from Handwriting using ML and CNN

## 📚 References
The handwriting traits and characteristics used in this project are derived from established literature:

- *The Graduate Course in Handwriting Analysis* by Handwriting Analysts International (HAI)
- *Personality Traits in Handwriting* by Ravindra Negi
- *Your Handwriting Can Change Your Life* by Vimala Rodgers

---

## 🧠 Project Overview

The proposed system is divided into two main components:

1. **Machine Learning Module**
2. **Deep Learning Module**

Python is used as the core programming language due to its rich ecosystem of open-source libraries for computer vision and deep learning.

---

## 🧮 Machine Learning Module

### OpenCV Preprocessing

- **Image Preprocessing**: Grayscale, binarization, bilateral filtering
- **Line Straightening**: Dilation, contouring, warp affine
- **Feature Extraction**: 
  - Baseline
  - Top margin
  - Line spacing
  - Letter size
  - Word spacing
  - Pen pressure
  - Slant angle

### Example: Letter Size Extraction

```python
# Pseudocode for estimating letter size
for each line in handwriting:
    if horizontal_projection > threshold:
        count += 1
letter_size = average(counts)
```

---

## 🧠 Deep Learning Module

### 📦 Dataset

- **Base Dataset**: [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- **Custom Dataset**: 1996 samples split into 11 categories based on 't' crossbar traits:
  1. Balanced crossbar
  2. Down slopping
  3. Flat crossbar
  4. High crossbar
  5. Left dominant crossbar
  6. Lengthy crossbar
  7. Low crossbar
  8. Medium length crossbar
  9. Right dominant crossbar
  10. Short length crossbar
  11. Up-slopping

---

## 🧠 Character Recognition CNN

### 📐 Architecture

- Input size: 28x28 grayscale image
- Conv Layer (32 filters) → ReLU → MaxPooling
- Conv Layer (64 filters) → ReLU → MaxPooling
- Dense Layer → Softmax Activation (26 classes)

### 🧪 Dataset
- **Training Dataset**: EMNIST Letters
- **Target Class**: Lowercase 't'

---

## 🧠 Trait Classification CNN

### 🧱 CNN Architecture

- **Conv2D → BatchNorm → MaxPooling → Dropout**
- **4 Conv layers**
- **3 Fully Connected Layers**
- Final layer uses **Sigmoid Activation** for binary classification

---

## 🔢 Feature Normalization

| Feature        | Values (Encoded) |
|----------------|------------------|
| Baseline       | 0 = Descending, 1 = Ascending, 2 = Straight |
| Top Margin     | 0 = Medium/Bigger, 1 = Narrow |
| Line Spacing   | 0 = Big, 1 = Small, 2 = Medium |
| Letter Size    | 0 = Big, 1 = Small, 2 = Medium |
| Word Spacing   | 0 = Big, 1 = Small, 2 = Medium |
| Pen Pressure   | 0 = Heavy, 1 = Light, 2 = Medium |
| Slant Angle    | 0–6 based on slant |

---

## ✂️ Segmentation Module

### 🧩 Steps
1. Read input image
2. Segment text lines
3. Segment words from lines
4. Segment characters using histogram and baseline detection
5. Normalize input characters to 28x28

---

## 📈 Performance Metrics

| Metric       | Character CNN | Trait CNN |
|--------------|----------------|-----------|
| **Accuracy** | 94.7%          | 95%       |
| **Precision**| –              | 93%       |
| **Recall**   | –              | 96.7%     |
| **F1 Score** | –              | 92%       |

---

## 📊 Evaluation Methods

- **Confusion Matrix**
- **Precision, Recall, F1 Score**
- **Validation Curves**

---

## 🧮 SVM Classifier

Support Vector Machines (SVMs) were also tested to classify binary traits using extracted features. Traits were encoded and trained using scikit-learn's SVM module for comparison.

---

## 🔧 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Matplotlib

---

## 📁 Folder Structure

```
project/
│
├── test_images/            # Raw handwriting samples
├── result/
│   ├── characters/         # Segmented characters
│   └── t_letters/          # Detected 't' letters
├── datasets/
│   ├── t_traits/           # 11 Trait folders
│   └── emnist/             # Base dataset
├── models/                 # Saved CNN models
├── scripts/
│   ├── segmentation.py
│   └── cnn_classifier.py
└── README.md               # You're here!
```

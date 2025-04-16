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

# Pseudocode for estimating letter size
for each line in handwriting:
    if horizontal_projection > threshold:
        count += 1
letter_size = average(counts)

# 🧠 Deep Learning Module

## 📦 Dataset

- **Base Dataset**: [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- **Custom Dataset**: 1996 samples categorized by 't' crossbar traits:
  - Balanced crossbar
  - Down slopping
  - Flat crossbar
  - High crossbar
  - Left dominant crossbar
  - Lengthy crossbar
  - Low crossbar
  - Medium length crossbar
  - Right dominant crossbar
  - Short length crossbar
  - Up-slopping

---

## 🧠 Character Recognition CNN

### 📐 Architecture

- Input: 28x28 grayscale image  
- Conv Layer (32 filters) → ReLU → MaxPooling  
- Conv Layer (64 filters) → ReLU → MaxPooling  
- Dense Layer → Softmax Activation (26 classes)

---

## 🧠 Trait Classification CNN

### 🧱 CNN Architecture

- Conv2D → BatchNorm → MaxPooling → Dropout  
- 4 Convolutional layers  
- 3 Fully Connected layers  
- Final layer: Sigmoid Activation (Binary Classification)

---

## 🔢 Feature Normalization

| Feature        | Encoded Values                                 |
|----------------|-------------------------------------------------|
| Baseline       | 0 = Descending, 1 = Ascending, 2 = Straight     |
| Top Margin     | 0 = Medium/Bigger, 1 = Narrow                   |
| Line Spacing   | 0 = Big, 1 = Small, 2 = Medium                  |
| Letter Size    | 0 = Big, 1 = Small, 2 = Medium                  |
| Word Spacing   | 0 = Big, 1 = Small, 2 = Medium                  |
| Pen Pressure   | 0 = Heavy, 1 = Light, 2 = Medium                |
| Slant Angle    | 0 to 6 based on slant                           |

---

## ✂️ Segmentation Module

### 🧩 Steps

1. Read input image  
2. Segment text lines  
3. Segment words from lines  
4. Segment characters using histogram and baseline detection  
5. Normalize characters to 28x28

---

## 📈 Performance Metrics

| Metric    | Character CNN | Trait CNN |
|-----------|----------------|-----------|
| Accuracy  | 94.7%          | 95%       |
| Precision | –              | 93%       |
| Recall    | –              | 96.7%     |
| F1 Score  | –              | 92%       |

---

## 📊 Evaluation Methods

- Confusion Matrix  
- Precision, Recall, F1 Score  
- Validation Curves

---

## 🧮 SVM Classifier

Support Vector Machines (SVMs) were used to classify binary traits from extracted features. Models were built and tested using scikit-learn.

---

## 🔧 Technologies Used

- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy / Pandas  
- Scikit-learn  
- Matplotlib



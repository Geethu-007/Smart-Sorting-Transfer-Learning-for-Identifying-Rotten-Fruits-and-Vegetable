# Smart-Sorting-Transfer-Learning-for-Identifying-Rotten-Fruits-and-Vegetable

This is the Dataset Link - https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification

# 🧠 Smart Sorting: Transfer Learning for Identifying Rotten Fruits & Vegetables

## 📌 Overview

Smart Sorting is a deep learning–based system designed to automatically classify **fresh and rotten fruits and vegetables** using **Transfer Learning**. The project aims to support:

* supermarkets
* warehouses
* food processing industries
* supply chain centers

by improving the **speed, accuracy, and efficiency** of quality inspection, ultimately reducing:

✅ food waste
✅ manual labor
✅ customer complaints
✅ operational costs

---

## 🎯 Problem Statement

Traditional fruit sorting methods are:

❌ manual and time-consuming
❌ subjective (depends on worker judgment)
❌ error-prone
❌ unsuitable for large-scale operations

Food spoilage leads to:

* economic losses
* health concerns
* product rejection during distribution

Thus, a reliable automated solution is required.

---

## ✅ Solution

This system uses **Transfer Learning with pre-trained CNN models** to classify produce as:

* Fresh
* Rotten

The workflow:

1. Capture or input an image
2. Preprocess the image
3. Run through trained classification model
4. Output freshness status

---

## 🧩 Why Transfer Learning?

Instead of training a CNN model from scratch, which requires:

* large datasets
* high computation
* long training time

Transfer Learning leverages models trained on ImageNet, such as:

* MobileNetV2
* ResNet50
* VGG16

These models have already learned:

✅ color patterns
✅ textures
✅ shapes
✅ edges

Only the final layers are retrained, making the system:

✅ faster to develop
✅ more accurate with small datasets

---

## 🏗️ System Architecture

```
Image Input
    ↓
Preprocessing (resize, normalization, augmentation)
    ↓
Pre-trained CNN (feature extractor)
    ↓
Custom Dense Layers
    ↓
Softmax Output
(Fresh / Rotten)
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib / Seaborn
* Transfer Learning (e.g., MobileNetV2 / ResNet50)

---

## 📁 Dataset

The dataset consists of images of fresh and rotten:

* apples
* bananas
* oranges
* tomatoes
* other 24 items 

Preprocessing includes:

* resizing to 224x224
* normalization
* augmentation (rotation, flipping, brightness)

---

## 🚀 Model Training

### Steps:

1️⃣ Freeze base model layers
2️⃣ Add custom classification head
3️⃣ Train only the new layers
4️⃣ Fine-tune upper CNN layers

### Evaluation Metrics:

✅ Accuracy
✅ Confusion Matrix
✅ Precision / Recall
✅ F1 Score

---

## 📊 Results

The model achieved:

✅ 95%+ accuracy on test data
✅ strong generalization to unseen images

The model successfully identifies:

* mold patterns
* color deterioration
* texture changes
* dark spot formation

---

## 🧪 How to Run the Project

```bash
git clone <repository_url>
cd Smart-Sorting
pip install -r requirements.txt
python app.py
```

---

## 🖥️ Deployment

This model can be deployed using:

* Flask
* Streamlit
* FastAPI
* Mobile / Edge Devices (Raspberry Pi)

---

## 🌟 Applications

✅ Automated sorting belts in industries
✅ Supermarket quality verification
✅ Warehouse monitoring systems
✅ Food packaging automation

---

## 🔮 Future Scope

* Multi-class classification (fresh / semi-rotten / rotten)
* Object detection to locate fruits in images
* Integration with robotic sorting arms
* Mobile app deployment
* IoT-based smart sorting machine

---

## 👨‍💻 Author

**Geethu-007**

---

## ⭐ Contribute

Pull requests are welcome! Feel free to:

* improve model performance
* extend dataset
* enhance deployment

---

## 📝 License

This project is open-source and available under the MIT License.

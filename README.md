# Digit Recognizer Project 🎯

Hello👋 This project was developed to classify handwritten digits using the Digit Recognizer Dataset from Kaggle. Trained with a deep learning-based Convolutional **Neural Network (CNN) model**, this project achieved a 98.82% accuracy rate on the validation set. The model's performance has also been tested with specially drawn images and detailed analyses have been performed.

---

## Technologies and Tools Used
- **Python 3.8+**
- **TensorFlow 2.0+**
- **Keras API**
- **OpenCV** (custom image processing)
- **Matplotlib** (graphical visualization)
- **Scikit-learn** (performance analysis)
  
---

## Project Summary

- **Objective:** Classifying handwritten digits using Kaggle's Digit Recognizer dataset.
- **Method:** Data processing, model training, and evaluation with a Convolutional Neural Network (CNN)-based model.
- **Result:** Training accuracy reached 99.00%, while validation accuracy was 98.82%. The model successfully made predictions on user-created custom images.

---

## Project Results

### 1. Training Process Results
The accuracy and loss values during the model's training process, summarized for each epoch, are shown in the table below. Throughout the training, the model's accuracy continuously improved, reaching a **99.00% accuracy rate** by the 5th epoch. The validation accuracy reached **98.82%.** The loss values also decreased steadily, demonstrating a successful learning process. The following visual shows the detailed training and validation results for each epoch:

![7](https://github.com/user-attachments/assets/4cbe210a-de84-4f37-9585-a908f1a1edbb)

### 2. Model Accuracy and Loss Graphs
Graphs of the accuracy and loss values obtained during the training and validation processes:  

![1](https://github.com/user-attachments/assets/016a8478-57c0-4a66-896a-90dbabeccf2d)

### 3. Validation Confusion Matrix
Doğrulama veri setinde modelin sınıflandırma başarısını gösteren matriks:

![2](https://github.com/user-attachments/assets/e0359142-8ef9-439b-a4e2-3e8a4bb37e5c)

### 4. Özel Resimlerden Tahmin Örnekleri
Matrix showing the model's classification success on the validation dataset:
- Prediction: 7 (Actual: 7)

  ![5](https://github.com/user-attachments/assets/e157f360-e0fe-423b-9dcc-406535a40ab5)

- Prediction: 0 (Actual: 2)
  
  ![4](https://github.com/user-attachments/assets/6d364096-4893-4aa0-b307-54fc12cd9e4c)
  
---

📂 **Report Contents**:
1. **Project Introduction:** The project’s objectives, methods used, and general approach.
2. **Technologies and Tools Used:** How the project was developed with technologies like Python, TensorFlow, OpenCV, Matplotlib, etc.
3. **Dataset and Preprocessing:** Preparing the data with the Kaggle dataset, normalization, data splitting, and reshaping.
4. **Model Architecture:** A detailed explanation of the CNN layers used in the deep learning model.
5. **Model Training:** Training process of the model, accuracy, and loss analysis per epoch.
6. **Validation Set Performance:** Performance of the model on the validation set and confusion matrix analysis.
7. **Testing with Custom Images:** Testing the model with custom hand-drawn images and reviewing prediction results.
8. **Conclusion and Future Work:** Overall evaluation of the project and recommendations for further improvements.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2 as cv

# Kaggle Veri Setini Yüklendi ve İşlendi
train = pd.read_csv('./data/train.csv')  # Eğitim veri seti
test = pd.read_csv('./data/test.csv')    # Test veri seti

# X (giriş verisi) ve Y (etiketler) olarak ayrıldı.
X = train.drop(columns=['label']).to_numpy()
Y = train['label'].to_numpy()
X_test = test.to_numpy()

# Normalizasyon ve Şekillendirme
X = X / 255.0  # Normalizasyon, piksel değerlerini [0-255] aralığından [0-1] aralığına normalleştirildi
X_test = X_test / 255.0
X = X.reshape(-1, 28, 28, 1)  # 28x28 boyutunda tek kanal (gri tonlamalı) olarak duzenlendi
X_test = X_test.reshape(-1, 28, 28, 1)

# Veriler, %80 eğitim ve %20 doğrulama seti olacak şekilde bölündü
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# CNN yani konvolüsyonel sinir agi modeli oluşturuldu.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)), # Giriş katmanı
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), # Konvolüsyon katmanı
    tf.keras.layers.MaxPooling2D(2, 2), # Maksimum havuzlama, pooling görüntünün boyutunu küçültür ve önemli bilgileri korur.
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(), # Veriyi düzleştirir
    tf.keras.layers.Dense(256, activation='relu'), # Tam bağlı katman, negatif değerleri sıfır yaparak doğrusal olmayan öğrenmeyi sağlar.
    tf.keras.layers.Dropout(0.5), # Eğitim sırasında rastgele nöronları devre dışı bırakarak aşırı öğrenmeyi önler
    tf.keras.layers.Dense(10, activation='softmax') # Çıkış katmanı
])

# model.compile, modeli eğitime hazırlamak için kullanılır.
# Modelin öğrenme sürecinde hangi optimizasyon algoritmasını, kayıp fonksiyonunu ve metriği kullanacağını belirler.
# Adam Optimizer: Modelin ağırlıklarını optimize eder
# Kayıp Fonksiyonu (Loss Function) modelin tahminleri ile gerçek değerler arasındaki farkı hesaplar.
# sparse_categorical_crossentropy, sınıf etiketlerinin integer (tam sayı) formatında olduğu durumlarda kullanılır.
# Modelin performansını değerlendirmek için doğruluk (accuracy) kullanılır.

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model egitildi
# Model 5 epoch boyunca eğitilir.
# Eğitim ve doğrulama verisi üzerinde doğruluk (accuracy) ve kayıp (loss) ölçülür.

history = model.fit(X_train, Y_train, epochs=5, batch_size=128, validation_data=(X_val, Y_val))

# Eğitim sonuçlarını görselleştirdim. pt.figure ile grafik figürünün boyutunu ayarladım. Genişlik 12 inç, yükseklik 5 inç
plt.figure(figsize=(12, 5))

# Doğruluk grafiği cizildi
plt.subplot(1, 2, 1) #Grafigin yerlesimi ayarlanir. Bir sayfa üzerinde 1 satır, 2 sütun grafik olmasini ve ilkinin doğruluk grafigi olacagini soyler
plt.plot(history.history['accuracy'], label='Train Accuracy') #Train Accuracy cizilir
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') #Validation Accuracy cizilir
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Kayıp grafiği cizildi
plt.subplot(1, 2, 2) #1 satir, 2 sutun grafikten 2. sutun olacagini soyler
plt.plot(history.history['loss'], label='Train Loss') #Train Loss cizilir
plt.plot(history.history['val_loss'], label='Validation Loss') #Validation Loss cizilir
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout() #Grafik düzenini otomatik olarak optimize eder, birden fazla alt grafik olduğunda çakışmaları önler.
plt.show() #Çizilen grafikleri ekrana görüntüler.

# Doğrulama seti için modelin tahmin ettiği etiketler ve gerçek etiketler karşılaştırılarak bir Confusion Matrix oluşturulur.
val_predictions = model.predict(X_val) #Model, doğrulama setindeki veriler (X_val) için tahminler yapar.
val_predicted_labels = np.argmax(val_predictions, axis=1) #Olasılık değerlerinden en yüksek olanını seçer ve buna karşılık gelen sınıfı (etiketi) döndürür.

conf_matrix = confusion_matrix(Y_val, val_predicted_labels, labels=np.arange(10))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.arange(10))
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix for Validation Set")
plt.show()

# Test veri seti uzerinde tahmin yapildi
test_predictions = model.predict(X_test)
predicted_labels = np.argmax(test_predictions, axis=1)

# Submission dosyasını oluşturdum
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_labels) + 1),
    "Label": predicted_labels
})
submission.to_csv('submission.csv', index=False)
print("Submission dosyası oluşturuldu: submission.csv")

# Ozel resimler uzerinde tahmin yapildi
image_folder = './custom_images/'
image_paths = [
    image_folder + '1.png',
    image_folder + '2.png',
    image_folder + '3.png',
    image_folder + '4.png',
    image_folder + '5.png'
]

# Çizdiğim rakamların gerçek etiketleri (örneğin 2, 3, 5, 7, 8)
true_labels = [2, 3, 5, 7, 8]

# Özel resimler işlendi (gri tonlama, yeniden boyutlandırma, normalizasyon).
# Model bu resimler üzerinde tahmin yapar ve tahminler görselleştirilir.
custom_images = []
for path in image_paths:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)  # Gri tonlamaya çevir
    if img is not None:
        img = cv.resize(img, (28, 28))  # 28x28 boyutuna küçült
        img = img / 255.0  # Normalize et
        img = img.reshape(28, 28, 1)
        custom_images.append(img)
    else:
        print(f"Resim yüklenemedi: {path}")

X_custom = np.array(custom_images)

# Tahmin yap
custom_predictions = model.predict(X_custom)
custom_predicted_labels = np.argmax(custom_predictions, axis=1)

# Ozel resimler için tahmin edilen ve gerçek etiketler karşılaştırılarak Confusion Matrix oluşturuldu

custom_conf_matrix = confusion_matrix(true_labels, custom_predicted_labels, labels=np.arange(10))
disp_custom = ConfusionMatrixDisplay(custom_conf_matrix, display_labels=np.arange(10))
disp_custom.plot(cmap='Blues')
plt.title("Confusion Matrix for Custom Images")
plt.show()

# Ozel resimlerin tahmin sonuçlarını görsellestirildi
for idx, pred in enumerate(custom_predictions):
    predicted_label = np.argmax(pred) #Modelin verdiği tahmin olasılıkları arasından en yüksek değeri bulur, buna karşılık gelen sınıfı (etiketi) döndürür.
    print(f"Image {idx + 1}: Predicted Label: {predicted_label}, True Label: {true_labels[idx]}")
    plt.imshow(X_custom[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {predicted_label} (True: {true_labels[idx]})")
    plt.xticks([])  # X ekseninde degerleri kaldırır
    plt.yticks([])  # Y ekseninde degerleri kaldırır
    plt.show()

# Model kaydedildi
model.save("digit_recognizer_model.h5")
print("Model saved as digit_recognizer_model.h5")






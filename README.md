# Digit Recognizer Projesi 🎯

Merhaba👋 Bu proje, Kaggle'daki [Digit Recognizer Dataset](https://www.kaggle.com/competitions/digit-recognizer/data) kullanılarak el yazısı rakamlarını sınıflandırmak amacıyla geliştirilmiştir. Derin öğrenme tabanlı **Convolutional Neural Network (CNN)** modeli ile eğitilen bu proje, doğrulama setinde %98.82 doğruluk oranına ulaşmıştır. Modelin performansı, özel olarak çizilmiş resimlerle de test edilmiş ve detaylı analizler gerçekleştirilmiştir.

📄 **Detaylı Proje Raporu**: Daha fazla bilgi almak için proje raporumu inceleyebilirsiniz. 
👉 [Raporu Görüntüle](https://github.com/EfsunPeynirci/Yuksek_Duzey_Programlama_Odevi_Digit_Recognizer/blob/main/Rapor.pdf)

---

## Kullanılan Teknolojiler ve Araçlar
- **Python 3.8+**
- **TensorFlow 2.0+**
- **Keras API**
- **OpenCV** (özel resim işleme)
- **Matplotlib** (grafik görselleştirme)
- **Scikit-learn** (performans analizi)

---

## Proje Özeti

- **Amaç:** Kaggle'ın Digit Recognizer veri setiyle el yazısı rakamların sınıflandırılması.
- **Yöntem:** Convolutional Neural Network (CNN) tabanlı bir model ile veri işleme, model eğitimi ve değerlendirme.
- **Sonuç:** Eğitim doğruluğu %99.00, doğrulama doğruluğu ise %98.82 olarak elde edilmiştir. Kullanıcı tarafından oluşturulan özel resimlerde model başarıyla tahminler gerçekleştirmiştir.

---

## Proje Sonuçları

### 1. Eğitim Süreci Sonuçları
Modelin eğitim sürecindeki doğruluk (accuracy) ve kayıp (loss) değerleri, her bir epoch için aşağıdaki tabloda özetlenmiştir. Eğitim boyunca modelin doğruluk oranı sürekli artmış ve 5. epoch sonunda **%99.00 doğruluk oranına** ulaşmıştır. Doğrulama doğruluğu (validation accuracy) ise **%98.82** olarak elde edilmiştir. Kayıp değerleri de düzenli olarak azalarak başarılı bir öğrenme sürecini göstermiştir. Aşağıdaki görsel, her epoch için eğitim ve doğrulama sonuçlarını detaylı olarak göstermektedir:

![7](https://github.com/user-attachments/assets/4cbe210a-de84-4f37-9585-a908f1a1edbb)

### 2. Model Accuracy ve Loss Grafikleri
Eğitim ve doğrulama süreçlerinde elde edilen doğruluk (accuracy) ve kayıp (loss) değerlerinin grafikleri:  

![1](https://github.com/user-attachments/assets/016a8478-57c0-4a66-896a-90dbabeccf2d)

### 3. Validation Confusion Matrix
Doğrulama veri setinde modelin sınıflandırma başarısını gösteren matriks:

![2](https://github.com/user-attachments/assets/e0359142-8ef9-439b-a4e2-3e8a4bb37e5c)

### 4. Özel Resimlerden Tahmin Örnekleri
Modelin özel resimlerde yaptığı tahminler:
- Tahmin: `7` (Gerçek: `7`)

  ![5](https://github.com/user-attachments/assets/e157f360-e0fe-423b-9dcc-406535a40ab5)

- Tahmin: `0` (Gerçek: `2`)
  
  ![4](https://github.com/user-attachments/assets/6d364096-4893-4aa0-b307-54fc12cd9e4c)
  
---

📄 **Detaylı Proje Raporu**: Daha fazla bilgi almak için proje raporumu inceleyebilirsiniz. 
👉 [Raporu Görüntüle](https://github.com/EfsunPeynirci/Yuksek_Duzey_Programlama_Odevi_Digit_Recognizer/blob/main/Rapor.pdf)

---

📂 **Rapor İçeriği**:
1. **Proje Tanıtımı:** Projenin amacı, kullanılan yöntemler ve genel yaklaşım.
2. **Kullanılan Teknolojiler ve Araçlar:** Python, TensorFlow, OpenCV, Matplotlib gibi teknolojilerle projenin nasıl geliştirildiği.
3. **Veri Seti ve Ön İşleme:** Kaggle veri seti ile verilerin hazırlanması, normalizasyon, veri ayrıştırma ve şekillendirme.
4. **Model Mimarisi:** Derin öğrenme modeli için kullanılan CNN katmanlarının detaylı açıklaması.
5. **Model Eğitimi:** Modelin eğitim süreci, epoch bazlı doğruluk ve kayıp analizleri.
6. **Validation Set Performansı:** Modelin doğrulama seti üzerindeki performansı ve confusion matrix analizi.
7. **Özel Resimlerle Test:** Modelin özel çizilen resimlerle test edilmesi ve tahmin sonuçlarının incelenmesi.
8. **Sonuç ve Gelecekteki Çalışmalar:** Projenin genel değerlendirmesi ve geliştirilmesi için öneriler.

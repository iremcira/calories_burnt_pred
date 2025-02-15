# Calories Burnt Prediction 🚴‍♂️🔥

## 📌 Project Overview
This project aims to predict the calories burnt during various physical activities using machine learning techniques. The model is trained using the **XGBRegressor** algorithm after performing extensive **data analysis, visualization, feature engineering, and model evaluation**.

---

## 🛠 Dependencies & Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
```

---

## 🔬 Data Processing Workflow
### 1️⃣ Importing the Dependencies
The necessary libraries are imported for data handling, visualization, and model training.

### 2️⃣ Data Collection & Processing
Two datasets (**exercise.csv** and **calories.csv**) are loaded using `pandas`, then merged based on a common column.

### 3️⃣ Exploratory Data Analysis (EDA)
- Basic information about the dataset (`.info()`, `.describe()`)
- Checking for missing values
- Encoding categorical features (Gender ➝ Numerical values)

### 4️⃣ Data Visualization 📊
#### **📌 Distribution Graph**
```python
sns.distplot(data['Calories'])
plt.title("Calories Distribution")
plt.show()
```
![Distribution Graph](images/dist_graph.png)

#### **📌 Correlation Heatmap**
```python
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
```
![Heatmap](images/heatmap.png)

### 5️⃣ Feature Engineering & Model Training
- **Separating Features & Target Variable** (Calories burnt)
- **Splitting Data** into Training & Testing sets
- **Training Model** using `XGBRegressor`

```python
model = XGBRegressor()
model.fit(X_train, Y_train)
```

### 6️⃣ Model Evaluation
- **Making Predictions** on test data
- **Calculating Mean Absolute Error (MAE)**

```python
y_pred = model.predict(X_test)
mae = mean_absolute_error(Y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
```

---

## 📈 Mathematical Background

The model is based on **Gradient Boosting**, which minimizes the loss function iteratively. The loss function used is Mean Absolute Error (MAE):


$$\left (  MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| \right)$$

where:
- $y_i$ is the actual value,
- $\hat{y}_i$ is the predicted value,
- $n$  is the total number of observations.

---

## 🚀 Possible Improvements
✅ **Feature Engineering**: Introducing additional activity parameters (e.g., heart rate, environmental conditions).  
✅ **Hyperparameter Tuning**: Optimizing XGBoost hyperparameters for better performance.  
✅ **Ensemble Methods**: Combining XGBoost with other regressors to improve accuracy.  

---

## 📜 Usage
1️⃣ Clone the repository:
```bash
git clone https://github.com/iremcira/calories_burnt_pred.git
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
3️⃣ Run the script:
```bash
python calories_prediction.py
```

---

## 🌍 Türkçe Açıklama

### 📌 Proje Özeti
Bu proje, farklı fiziksel aktiviteler sırasında yakılan kalorileri tahmin etmek için makine öğrenmesi tekniklerini kullanmaktadır. Model, **XGBRegressor** algoritması ile eğitilmiştir ve **veri analizi, görselleştirme, özellik mühendisliği ve model değerlendirme** aşamalarından geçmiştir.

### 🛠 Kullanılan Kütüphaneler

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
```

### 🔬 Veri İşleme Süreci
1️⃣ **Gerekli kütüphaneler içe aktarılır.**
2️⃣ **İki farklı veri seti (exercise.csv ve calories.csv) birleştirilir.**
3️⃣ **Eksik veriler kontrol edilir ve temizlenir.**
4️⃣ **Veri görselleştirme yapılır (Dağılım grafikleri, Isı haritaları).**
5️⃣ **Bağımsız değişkenler ve hedef değişken ayrılır.**
6️⃣ **Veri eğitim ve test kümelerine bölünür.**
7️⃣ **XGBRegressor kullanılarak model eğitilir.**
8️⃣ **Test verileri üzerinde tahminler yapılır ve MAE hesaplanır.**

### 📈 Matematiksel Arka Plan

XGBRegressor, **Gradient Boosting** prensibine dayalıdır ve **Mean Absolute Error (MAE)** metriğini minimize etmeyi hedefler:

$$\left (  MAE = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| \right)$$

### 🚀 Olası Geliştirmeler
✅ **Daha fazla özellik eklenebilir (nabız, ortam sıcaklığı gibi).**  
✅ **Hiperparametre optimizasyonu ile model daha iyi hale getirilebilir.**  
✅ **Diğer regresyon modelleri ile birleştirilerek doğruluk artırılabilir.**  

### 📜 Kullanım
1️⃣ Repository’i klonlayın:
```bash
git clone https://github.com/iremcira/calories_burnt_pred.git
```
2️⃣ Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```
3️⃣ Script’i çalıştırın:
```bash
python calories_prediction.py
```

---


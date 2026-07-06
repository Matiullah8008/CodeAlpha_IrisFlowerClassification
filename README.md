# 🌸 Iris Flower Classification using Machine Learning

### CodeAlpha Data Science Internship - Task 1

This project aims to build a machine learning model that classifies Iris flowers into three species: **Setosa, Versicolor,** and **Virginica** based on their sepal and petal measurements. The project covers data exploration, preprocessing, model training, evaluation, and prediction.

---

## 📁 Project Structure
- `Task1_Iris_Flower_Classification.ipynb`: Jupyter notebook containing the complete pipeline (data analysis, model training, evaluation, and prediction).
- `iris_model.pkl`: The trained Random Forest classifier model saved using joblib.
- `Iris.csv`: The Iris dataset.
- `requirements.txt`: Python package dependencies.
- `Report.docx`: Microsoft Word version of the project report.

---

## 🚀 Installation & Setup

1. **Clone or download** this repository folder.
2. **Install the required packages** using `pip`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Jupyter Notebook** to view or run the classification pipeline:
   ```bash
   jupyter notebook Task1_Iris_Flower_Classification.ipynb
   ```

---

## 📊 Project Report & Methodology

### 1. Executive Summary
The objective of this project is to build an automated machine learning model capable of classifying Iris flower samples into one of three species:
* **Iris-setosa**
* **Iris-versicolor**
* **Iris-virginica**

Using a dataset of 150 samples with four physical features (sepal length, sepal width, petal length, and petal width), we developed a **Random Forest Classifier** that successfully classifies the species with **100% accuracy** on the test dataset.

### 2. Methodology & Steps Taken

#### Step 1: Data Exploration & Understanding
We loaded the dataset `Iris.csv` using Pandas. Initial checks confirmed:
* **Total Records:** 150 rows.
* **Total Columns:** 6 columns (`Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`).
* **Missing Values:** Zero null values, representing a clean and complete dataset.

#### Step 2: Data Preprocessing
* **ID Drop:** The `Id` column was dropped because it acts as a unique row counter and does not hold any predictive power.
* **Label Encoding:** The categorical target variable `Species` was converted into numerical categories using Scikit-learn's `LabelEncoder` (Setosa $\rightarrow$ 0, Versicolor $\rightarrow$ 1, Virginica $\rightarrow$ 2).

#### Step 3: Exploratory Data Analysis (EDA)
* **Distribution Check:** A count plot verified that the classes are perfectly balanced (50 samples for each of the three species).
* **Correlation Analysis:** A heatmap of correlation values showed a very strong positive correlation between `PetalLengthCm` and `PetalWidthCm` ($r = 0.96$).
* **Pairwise Inspection:** A pairplot highlighted that `Iris-setosa` is completely and linearly separable from the other two species based on its smaller petal length and width.

#### Step 4: Model Development
* **Train-Test Split:** The dataset was split into **80% training data** (120 samples) and **20% testing data** (30 samples) using a random state seed of 42 to ensure repeatability.
* **Algorithm Selection:** A **Random Forest Classifier** was chosen due to its high accuracy, stability, and resistance to overfitting on small tabular datasets.
* **Model Training:** The classifier was fitted onto the training features (`X_train`) and labels (`y_train`).

#### Step 5: Model Serialization
* The trained model was serialized using the `joblib` library and saved as `iris_model.pkl` for quick inference in the future without retraining.

---

## 📈 Results & Evaluation

### 3.1. Accuracy Score
The model was tested against the 30 unseen samples in the test set, achieving a perfect classification accuracy of **100.00%**.

### 3.2. Detailed Classification Metrics
The precision, recall, and F1-scores for all three species were evaluated:

| Species Name | Class Label | Precision | Recall | F1-Score | Test Samples (Support) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Iris-setosa** | 0 | 1.00 | 1.00 | 1.00 | 10 |
| **Iris-versicolor** | 1 | 1.00 | 1.00 | 1.00 | 9 |
| **Iris-virginica** | 2 | 1.00 | 1.00 | 1.00 | 11 |

* **Precision (1.00):** 100% of the model's species predictions were correct.
* **Recall (1.00):** The model successfully identified 100% of the actual instances of each species.
* **F1-Score (1.00):** The harmonic mean of precision and recall shows perfect model stability.

### 3.3. Confusion Matrix Breakdown
The test set results were distributed as follows:
* **Setosa:** 10 samples correctly predicted, 0 errors.
* **Versicolor:** 9 samples correctly predicted, 0 errors.
* **Virginica:** 11 samples correctly predicted, 0 errors.

---

## 💡 Key Findings & Discoveries

1. **Petals are Highly Informative:**
   * Petal length and petal width are the most crucial features for identifying the flower species.
   * `Iris-setosa` has significantly smaller petals (ranging between $1.0\text{ cm}$ to $1.9\text{ cm}$ in length) than the other two species.
2. **Species Boundary Separability:**
   * While `Iris-setosa` is easily distinguishable and isolated, `Iris-versicolor` and `Iris-virginica` exhibit slight overlaps in their sepal sizes, but remain highly separable using their petal sizes.
3. **Correlation:**
   * Flower petal length and petal width grow proportionally ($96\%$ correlation rate). Knowing one is an excellent indicator of the other.

---

## 💻 Running the Model (Inference Example)

To use the saved model to classify a new flower:

```python
import joblib

# Load the saved model
model = joblib.load("iris_model.pkl")

# New flower data: [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]
new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Predict
predicted_class = model.predict(new_flower)[0]
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

print("Predicted Species:", species_map[predicted_class])
# Output: Predicted Species: Iris-setosa
```

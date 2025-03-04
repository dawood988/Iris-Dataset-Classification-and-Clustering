# Iris Dataset Classification and Clustering

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Project Workflow](#project-workflow)
   - [Data Preprocessing & Exploratory Data Analysis (EDA)](#1-data-preprocessing--exploratory-data-analysis-eda)
   - [Feature Scaling & Encoding](#2-feature-scaling--encoding)
   - [Model Training & Evaluation](#3-model-training--evaluation)
   - [Model Optimization](#4-model-optimization)
4. [Installation & Dependencies](#installation--dependencies)
5. [Running the Project](#running-the-project)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Enhancements](#future-enhancements)
9. [Author](#author)
10. [License](#license)

## Overview
This project focuses on classifying the Iris dataset using various machine learning algorithms and performing clustering with K-Means. The goal is to compare the performance of different models, including logistic regression, KNN, decision trees, random forests, SVM, and artificial neural networks (ANN). Additionally, exploratory data analysis (EDA) and visualization techniques are employed to understand the dataset better.

## Dataset
The dataset used in this project is the **Iris dataset**, which contains 150 samples of iris flowers, categorized into three species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

### Features:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (target variable)

## Project Workflow

### 1. Data Preprocessing & Exploratory Data Analysis (EDA)
- Load the dataset using Pandas.
- Check for missing values.
- Perform descriptive statistics.
- Visualize the data using scatter plots, pie charts, and 3D plots.

### 2. Feature Scaling & Encoding
- Normalize numerical features using **MinMaxScaler**.
- Encode categorical target variables using **LabelEncoder**.

### 3. Model Training & Evaluation
Various classification models are implemented and evaluated:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)** (Hyperparameter tuning included)
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVM)**
- **Artificial Neural Networks (ANN) using TensorFlow/Keras**
- **K-Means Clustering** (to group similar observations together)

Each model is evaluated based on:
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-Score

### 4. Model Optimization
- **GridSearchCV** is used for hyperparameter tuning.
- **Cross-validation** is performed to ensure generalization.

## Installation & Dependencies
### Prerequisites:
Ensure you have the following libraries installed:
```sh
pip install numpy pandas matplotlib seaborn plotly scikit-learn tensorflow
```

## Running the Project
1. Clone this repository:
   ```sh
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```sh
   cd iris-classification
   ```
3. Run the script:
   ```sh
   python iris_classification.py
   ```

## Results
- **Visualization:** Scatter plots, heatmaps, and elbow method for clustering.
- **Model Performance:** The best performing model is selected based on accuracy and generalization ability.
- **ANN Performance:** A deep learning model is trained and compared with traditional ML models.

## Conclusion
This project demonstrates how different machine learning algorithms can be applied to a simple classification problem. Feature scaling, hyperparameter tuning, and cross-validation help improve model performance. The results indicate that some models perform better than others, depending on dataset characteristics.

## Future Enhancements
- Implement **XGBoost** and **LightGBM** for improved accuracy.
- Use **Principal Component Analysis (PCA)** for dimensionality reduction.
- Deploy the model using Flask or Streamlit.

## Author
- **Dawood M D**  

## License
This project is open-source and available under the [MIT License](LICENSE).


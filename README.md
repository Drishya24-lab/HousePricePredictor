# ML-Projects
Machine Learning Projects
# House Price Prediction Project

This project uses machine learning algorithms to predict house prices based on various features such as location, area size, number of bedrooms, and more. It leverages popular machine learning libraries and evaluates multiple regression models to select the best-performing one.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation and Usage](#installation-and-usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Model Saving](#model-saving)

## Project Overview
The goal of this project is to develop a regression model that can accurately predict house prices. The dataset contains information on properties, including location, area size, number of bedrooms, and property type. The project was implemented on Google Colab.

## Data Preprocessing
Steps for data preprocessing include:
- **Dropping Irrelevant Columns**: Removed columns that do not provide significant information for price prediction.
- **Handling Missing Values**: Checked for and handled any missing values in the dataset.
- **Encoding Categorical Variables**: Applied One-Hot Encoding to categorical variables like city, property type, province, and purpose.
- **Scaling**: Used `StandardScaler` to normalize numerical features for better model performance.

## Exploratory Data Analysis (EDA)
Performed exploratory analysis to understand relationships in the data, including:
- **Property Type vs. Price**
- **Province-wise Price Distribution**
- **Purpose vs. Price**
- Visualized the price distribution, relationships between property type and price, and the area size to gain insights for model building.

## Feature Engineering
- **Encoding**: Used OneHotEncoder to encode categorical columns to make them machine-readable.
- **Scaling**: Applied scaling to numeric features to standardize the data, ensuring consistent feature magnitudes.

## Modeling
Multiple regression models were evaluated to predict house prices, including:
1. **Linear Regression**
2. **Decision Tree Regressor**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **AdaBoost Regressor**
6. **XGBoost Regressor**

## Hyperparameter Tuning
Utilized `GridSearchCV` for cross-validation and hyperparameter tuning to enhance the performance of the XGBoost model. Optimal parameters were selected based on performance metrics, and the tuned model was saved for future use.

## Evaluation Metrics
The following metrics were used to evaluate model performance:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-Squared (R2)**

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Drishya24-lab/ML-Projects.git
### Running the Project
1. Open the Jupyter notebook (.ipynb) file on Google Colab.
2. Execute the cells in order.

### Results
-Mean Absolute Error: 0.17841783631324432
-Mean Squared Error: 0.3730125534748223
-R-Squared: 0.5987740410060091
### Future Improvements
- Experiment with advanced models like neural networks.
- Implement deployment using Flask or FastAPI.
- Expand the dataset for better model robustness.

### Model Saving
The final model was saved using joblib for future use:

```python
joblib.dump(final_model, '/content/drive/MyDrive/final_model.pkl')

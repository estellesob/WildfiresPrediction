# Wildfire Risk Prediction

## Project Overview

This project aims to predict wildfire risk by analyzing meteorological data from the US Wildfire Dataset (2014–2025). The primary objective is to build a binary classification model to distinguish between "Fire" and "No Fire" events.
A significant focus of this work is Green AI, prioritizing models that are computationally efficient and simple while maintaining predictive performance.

## Dataset

The analysis is based on the US Wildfire Dataset, which provides historical weather and fire occurrence data.

*Source:* Kaggle - US Wildfire Dataset : https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset/data

*Target Variable:* Wildfire (Binary classification: "Yes" or "No")

*Key Features:*
- tmmn / tmmx: Minimum/Maximum Temperature
- rmin / rmax: Minimum/Maximum Relative Humidity
- pr: Precipitation
- vs: Wind Speed
- vpd: Vapor Pressure Deficit
- erc, etr, bi: Fire indices like Energy Release Component
- latitude / longitude: Geospatial information

## Methodology

**1. Preprocessing**

The data preprocessing phase focused on cleaning and formatting the dataset for analysis. This included removing duplicate entries and filtering out anomalous values (specifically 32767, which indicates sensor errors). Additionally, the datetime column was converted into a proper datetime object to facilitate temporal analysis.

**2. Exploratory Data Analysis (EDA)**

An extensive exploratory analysis was conducted to understand the data's characteristics.

- Target Distribution: The dataset is highly imbalanced, with the vast majority of samples representing "No Fire" conditions.
- Correlation Analysis: A heatmap revealed multicollinearity among several weather variables, such as strong correlations between temperature and vapor pressure deficit.
- Geographical Distribution: A scatter plot of fire occurrences highlighted spatial clusters of wildfire events across the US.
- Feature Distribution: Histograms and boxplots were used to examine the distribution of key meteorological features like temperature and wind speed.

**3. Modeling**

Two supervised learning models were trained and evaluated. To address the class imbalance issue, both models were trained with class_weight="balanced".

- Logistic Regression: Used as a baseline linear model.
- Random Forest Classifier: Employed to capture non-linear relationships and complex interactions between weather variables.

## Limitations

Several constraints were identified during the project that impacted model performance:
- Dataset Size: The original dataset was too large for the available computational resources, necessitating subsampling to make the analysis feasible.
- Class Imbalance: The dataset is heavily imbalanced, with "No Fire" instances accounting for 95% of the data and "Fire" instances only 5%. This skew makes it challenging for models to accurately identify fire events without generating excessive false positives.
- Hardware Constraints: Training complex models on the full dataset would require GPU acceleration, which was not utilized in this implementation.

## Results Analysis

Both models demonstrated limitations in predictive capability, particularly regarding precision for the minority class:
- Low Precision: The precision for detecting fires was low across both models, indicating a high rate of false alarms.
- False Alarms: The models frequently flagged non-fire conditions as fire risks, a common issue when dealing with highly imbalanced datasets where the model tends to bias towards the majority class.
- Artificial Accuracy: While accuracy scores might appear reasonable (57% for LR and 69% for RF), they can be misleading in this context; evaluation metrics like F1-score and Precision-Recall AUC provide a more realistic view of performance on the minority class.

## Future Improvements

To enhance model performance, future work could include:
- XGBoost: Implementing gradient boosting algorithms like XGBoost could potentially improve predictive accuracy.
- Deep Learning: Neural networks might be better suited to capture complex, high-dimensional patterns in the data.
- Feature Engineering: Developing new features based on temporal trends or interactions between weather variables could provide better signals for the models.

## Requirements

The following Python libraries are required to run this project:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- kagglehub (for automated dataset download)

## Usage

- Clone this repository.
- Install the necessary dependencies.
- Run the Jupyter Notebook WildfireProject.ipynb.

The dataset will be automatically downloaded via kagglehub when the notebook is executed.

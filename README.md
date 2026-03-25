# GradeFlow

## About
This project implements an **Ordinal Logistic Regression** model to predict student performance levels (**Low**, **Medium**, **High**) based on various academic and lifestyle factors.  
It includes **data preprocessing**, **outlier detection**, **model training**, and **evaluation** using confusion matrices and classification reports to ensure robust predictive analysis.

---

## Files
- `ordinal-logistic-regression.py` → Python script implementing the model.  
- `student_performance.csv` → Dataset containing student information and performance levels.

---

## Steps Included

### 1️⃣ Data Preprocessing
- Loaded and cleaned the dataset using `pandas`.  
- Selected relevant features:  
  `['Age', 'Hours_Studied_Per_Week', 'Attendance_Rate', 'Past_GPA', 'Parent_Income', 'Extra_Curricular_Hours', 'Sleep_Hours']`
- Converted target labels from text to ordered numeric categories:  
  `{'Low': 0, 'Medium': 1, 'High': 2}`  
- Applied **log transformation** to handle outliers in numeric features.

---

### 2️⃣ Outlier Detection
- Visualized outliers using **Boxplots** and **Seaborn** heatmaps.  

---

### 3️⃣ Model Training
- Built an **Ordinal Logistic Regression** model using `OrderedModel` from `statsmodels`.  

---


## How to Run

1- Install Dependencies:
  ```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
```

2-Run :

  ```bash
python ordinal-logistic-regression.py
```

3- Model Evaluation :

- Boxplots (Before Outlier Handling) : Displays boxplots for all numerical features to show data distribution and detect outliers. Helps identify extreme or abnormal values in the dataset.


<p align="center">
<img width="1854" height="1013" alt="image" src="https://github.com/user-attachments/assets/a3c1c37f-7806-429f-9584-6b18dfa25baa" />
</p>


- Boxplots (After Log Transformation) : Shows boxplots again after applying `np.log1p()` transformation. Demonstrates how outlier influence is reduced, making the data more balanced.


<p align="center">
<img width="1855" height="984" alt="image" src="https://github.com/user-attachments/assets/91274b8d-9503-49b7-9637-0ff68e55082e" />
</p>


- Confusion Matrix : Visual representation of correct vs. incorrect predictions. Shows how well the model classified each student performance level (**Low**, **Medium**, **High**).


<p align="center">
<img width="1130" height="953" alt="image" src="https://github.com/user-attachments/assets/df83ba11-42fb-461a-b496-452ffcc41456" />
</p>


- Classification Report Heatmap : A heatmap of metrics including **Precision**, **Recall**, and **F1-Score** for each performance class. Provides a clear summary of how accurate and reliable the model is across all categories.


<p align="center">
<img width="1741" height="1039" alt="image" src="https://github.com/user-attachments/assets/9acfd449-8232-4be7-965d-4ef9e329b9df" />
</p>


 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.

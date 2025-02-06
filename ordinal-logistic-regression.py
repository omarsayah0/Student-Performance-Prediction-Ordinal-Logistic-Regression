import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as  np
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , classification_report 

def load_data():
    df = pd.read_csv("student_performance.csv")
    features = ['Age' ,'Hours_Studied_Per_Week' ,'Attendance_Rate','Past_GPA',  'Parent_Income','Extra_Curricular_Hours',  'Sleep_Hours']
    df['Final_Performance'] = df['Final_Performance'].map({'Low':0 , 'Medium':1 , 'High':2})
    x = df[features]
    order_quality = sorted(df['Final_Performance'].unique())
    df['Final_Performance'] = pd.Categorical(df['Final_Performance'] , categories=order_quality , ordered=True)
    y = df['Final_Performance']
    return (x , y)

def split_data(x , y):
    x_train , x_test , y_train , y_test = train_test_split(
        x , y , test_size= 0.2 , random_state=42
    )
    return (x_train , x_test , y_train , y_test)

def set_model(x , y):  
    model = OrderedModel(y , x , distr='logit')
    result = model.fit(method='bfgs' , disp=False)
    return (result)

def deal_outlier_show(x):
    features = ['Age' ,'Hours_Studied_Per_Week' ,'Attendance_Rate','Past_GPA',  'Parent_Income','Extra_Curricular_Hours',  'Sleep_Hours']
    x[['Age' ,'Hours_Studied_Per_Week' ,'Attendance_Rate','Past_GPA',  'Parent_Income','Extra_Curricular_Hours',  'Sleep_Hours']] =np.log1p(x[['Age' ,'Hours_Studied_Per_Week' ,'Attendance_Rate','Past_GPA',  'Parent_Income','Extra_Curricular_Hours',  'Sleep_Hours']])  
    figure_data = x.melt(value_vars=features , var_name='features' , value_name='value')
    plt.figure(figsize=(12 , 6))
    sns.boxplot(x = 'features' , y= 'value' , data=figure_data)
    plt.title("The outliers")
    plt.show()
    return (x)

def the_outliers(x):
    _, axes = plt.subplots(2, 4, figsize=(15, 8))  
    axes = axes.flatten()  
    features = ['Age' ,'Hours_Studied_Per_Week' ,'Attendance_Rate','Past_GPA',  'Parent_Income','Extra_Curricular_Hours',  'Sleep_Hours' ]
    for i, col in enumerate(features):
        sns.boxplot(y=x[col], ax=axes[i])
        axes[i].set_title(col)
    plt.tight_layout()
    plt.show()

def conf_matrix(y_test,y_pred):
    conf_matr = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matr)
    disp.plot(cmap='Blues')
    plt.show()

def class_report(y_test , y_pred):
    report = classification_report(y_test, y_pred, output_dict=True , zero_division=1)  
    df_report = pd.DataFrame(report).transpose()  
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True,  fmt=".2f", cmap="Blues",)
    plt.title("Classification Report Heatmap")
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    plt.show()

def main():
    x , y = load_data()
    deal_outlier_show(x)
    the_outliers(x)
    x_train , x_test , y_train , y_test =split_data(x , y)
    model = set_model(x_train , y_train)
    y_pred = model.predict(x_test)
    predicted_codes  = y_pred.idxmax(axis=1)
    predicted_class = predicted_codes.map(lambda code: y_train.cat.categories[code])
    conf_matrix(y_test ,predicted_class )
    class_report(y_test ,predicted_class )

if __name__ == "__main__":
    main()
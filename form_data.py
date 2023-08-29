
'''
age - age in years
sex - (1 = male; 0 = female)
cp - 가슴 통증(1) chest pain type
trestbps - 혈압 resting blood pressure (in mm Hg on admission to the hospital)
chol - 총 콜레스트롤 serum cholestoral in mg/dl
fbs - (공복 혈당 1 = 당뇨, 0 = 정상 fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - 심전도 검사( 1 = 이상, 0 = 정상 ) resting electrocardiographic results
thalach -  최대 심박수 maximum heart rate achieved
exang - 운동에 의한 협심증(1) exercise induced angina (1 = yes; 0 = no)
oldpeak - 휴식 대비 운동으로 인한 ST 분절 저하도 ST depression induced by exercise relative to rest
slope - 활동 ST 분절 피크의 기울기 the slope of the peak exercise ST segment
ca - 착색된 주요 혈관 수 number of major vessels (0-3) colored by flourosopy
thal - 3 = normal(정상); 6 = fixed defect(변화하지 않는 결함); 7 = reversable defect(되돌릴 수 없는 결함)
target - 정상(1), 비정상(0) have disease or not (1=yes, 0=no)
'''

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

heart_csv = "./heart.csv"
test_size = 0.25

def get_df():
    df = pd.read_csv(heart_csv).dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def get_normalized(df):
    scaler = MinMaxScaler()
    df[['age', 'trestbps','chol', 'thal', 'thalach']] = scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thal', 'thalach']])
    return df

def get_dataset(x, y):
    return train_test_split(x, y, test_size=test_size, random_state=42) #x_train, x_test, y_train, y_test



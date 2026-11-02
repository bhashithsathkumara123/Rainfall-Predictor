import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/weatherAUS.csv')

data = data.dropna(subset=['RainTomorrow'])

numeric_cols= data.select_dtypes(include=['int64','float64']).columns
for col in numeric_cols:
    data[col]= data[col].fillna(data[col].median())

text_cols= data.select_dtypes(include=['object']).columns
for col in text_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data['RainTomorrow']= data['RainTomorrow'].map({'Yes':1, 'No':0})
print(data) 

encoders = {}
cat_cols = ['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le  


features=['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine','WindGustSpeed', 'Humidity3pm', 'Pressure3pm',
         'Temp3pm','RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']

X=data[features]
y_class= data['RainTomorrow']
y_regr= data['Rainfall']

#classification
X_train_class, X_test_class, y_train_class, y_test_class= train_test_split(X,y_class,test_size=0.2,random_state=42)

#regression
rain_data = data[data['Rainfall'] == 1]
X_rain = rain_data[features]
y_rain = rain_data['Rainfall']

#Building classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_class, y_train_class)

#Building regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_rain, y_rain)

y_pred_class= classifier.predict(X_test_class)

print("Classification accuracy:", accuracy_score(y_test_class,y_pred_class))
print("\nClassification report:\n", classification_report(y_test_class,y_pred_class))

cm=confusion_matrix(y_test_class,y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


def hybrid_rain_predict(input_data):
    df_input = pd.DataFrame([input_data])
    
    # Use stored encoders safely
    for col in ['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm']:
        le = encoders[col]
        try:
            df_input[col] = le.transform(df_input[col])
        except ValueError:
            df_input[col] = -1  # Use -1 or a custom fallback
            print(f"Warning: {col} has unseen category. Assigned as -1.")

    rain_pred = classifier.predict(df_input)

    if rain_pred[0] == 1:
        rain_amount = regressor.predict(df_input)
        return f"Yes, it will rain tomorrow. Estimated rainfall: {rain_amount[0]:.2f} mm."
    else:
        return "No, it will not rain tomorrow."


example_input = {
    'MinTemp': 20.5,
    'MaxTemp': 32.0,
    'Rainfall': 5.0,
    'Evaporation': 4.0,
    'Sunshine': 8.0,
    'WindGustSpeed': 30,
    'Humidity3pm': 75,
    'Pressure3pm': 1012.0,
    'Temp3pm': 29.5,
    'RainToday': 'Yes',       
    'WindGustDir': 'NE',       
    'WindDir9am': 'N',
    'WindDir3pm': 'SE'
}


result = hybrid_rain_predict(example_input)
print("\nHybrid Model Output:")
print(result)
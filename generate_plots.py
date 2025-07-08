
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Preprocessing
df = df.drop(['customerID'], axis=1)
df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.dropna(inplace=True)

# Save Churn Distribution plot
plt.figure(figsize=(6, 6))
labels = ['No', 'Yes']
colors = ['#ff9999','#66b3ff']
explode = (0.1, 0)
df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=labels, colors=colors, explode=explode)
plt.title('Churn Distribution')
plt.ylabel('')
plt.savefig('churn_distribution.png')
plt.close()

# Label Encoding
le = LabelEncoder()
# Create a copy to avoid SettingWithCopyWarning
df_encoded = df.copy()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        df_encoded[column] = le.fit_transform(df_encoded[column])

# Train/Test Split
x = df_encoded.drop(['Churn'], axis=1)
y = df_encoded['Churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model Training and Prediction
cat_model = CatBoostClassifier(verbose=0)
cat_model.fit(x_train, y_train)
y_pred_cat = cat_model.predict(x_test)

# Save Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred_cat)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - CatBoost')
plt.savefig('confusion_matrix.png')
plt.close()

print('Images generated successfully.')

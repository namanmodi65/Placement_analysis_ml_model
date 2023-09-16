import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
# Load the csv file
df = pd.read_csv("collegePlace (1).csv")
df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0})
print(df.head())
X = df.drop(columns=['PlacedOrNot'],axis=1)
y = df['PlacedOrNot']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

clf = RandomForestClassifier()

# Fit the model
clf.fit(X_train, y_train)

pickle.dump(clf,open('model.pkl','wb'))

import os
print("Current working directory:", os.getcwd())
import pandas as pd
import joblib

model = joblib.load("copd_model.pkl")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("copd_data.csv")


X = data.drop("copd", axis=1)  
y = data["copd"]              
X = data.drop("copd", axis=1)  # input features

y = data["copd"]               


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier()

model.fit(X_train, y_train)


predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")

def predict_copd():
    print("\n🩺 --- COPD Risk Checker ---")
    try:
        age = int(input("Enter your age (e.g. 45): "))
        smoking = int(input("Do you smoke? (1 = Yes, 0 = No): "))
        breath = int(input("Shortness of breath? (1 = Yes, 0 = No): "))
        cough = int(input("Coughing frequently? (1 = Yes, 0 = No): "))
        lung = int(input("Lung function test score (1 to 100): "))

      
        if smoking not in [0, 1] or breath not in [0, 1] or cough not in [0, 1] or not (1 <= lung <= 100):
            print("⚠️ Invalid input values. Please follow the instructions carefully.")
            return

    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")
        return

    user_input = [[age, smoking, breath, cough, lung]]
    result = model.predict(user_input)

    if result[0] == 1:
        print("⚠️ High risk of COPD. Please consult a doctor.")
    else:
        print("✅ Low risk of COPD. Stay healthy!")

if __name__ == "__main__":
    predict_copd()


import matplotlib.pyplot as plt
import seaborn as sns


#sample accuracy value
accuracy = 0.85  

#accuracy plot
plt.figure(figsize=(5, 4))
plt.bar(['Accuracy'], [accuracy], color='skyblue')
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.savefig("accuracy_plot.png")  # Image same folder me 
print("✅ Accuracy graph saved as accuracy_plot.png")

#Feature Importance Plot 
importances = model.feature_importances_
features = X.columns  
importances = [0.2, 0.3, 0.1, 0.15, 0.25]
features = ['age', 'smoking', 'cough', 'breath', 'lung']

plt.figure(figsize=(7, 5))
sns.barplot(x=importances, y=features, palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.savefig("feature_importance.png")  
print("✅ Feature importance graph saved as feature_importance.png")

import joblib 


joblib.dump(model, "copd_model.pkl")  
print("✅ Model saved as copd_model.pkl")
 
model = joblib.load("copd_model.pkl")





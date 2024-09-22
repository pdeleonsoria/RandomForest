from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pickle import dump

train_data = pd.read_csv("https://raw.githubusercontent.com/pdeleonsoria/Decission-tree/main/data/processed/clean_train.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/pdeleonsoria/Decission-tree/main/data/processed/clean_test.csv")

train_data.head()
test_data.head()

#SEPARAR EN TRAIN Y TEST 
X_train = train_data.drop(["Outcome"], axis = 1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis = 1)
y_test = test_data["Outcome"]

#GENERAR RANDOM FOREST RandomForestClassifier(n_estimators= , max_depth= , random_state=42)

n_estimators_valores = [10, 25, 40, 55, 70, 85, 100] 
max_depth_valores = [1, 5, 10,15, 20]    

resultados = []

for n_estimators in n_estimators_valores:
    for max_depth in max_depth_valores:
       
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        exactitud = accuracy_score(y_test, y_pred)
        resultados.append((n_estimators, max_depth, exactitud))
        
resultados_df = pd.DataFrame(resultados, columns=["n_estimators", "max_depth", "exactitud",])

print (resultados_df)

md1 = resultados_df[resultados_df["max_depth"] == 1]
md5 = resultados_df[resultados_df["max_depth"] == 5]
md10 = resultados_df[resultados_df["max_depth"] == 10]
md15 = resultados_df[resultados_df["max_depth"] == 15]
md20 = resultados_df[resultados_df["max_depth"] == 20]

plt.figure(figsize=(10, 5))
plt.plot(md1["n_estimators"], md1["exactitud"], marker="o", label="Max Depth = 1")
plt.plot(md5["n_estimators"], md5["exactitud"], marker="o", label="Max Depth = 5")
plt.plot(md10["n_estimators"], md10["exactitud"], marker="o", label="Max Depth = 10")
plt.plot(md15["n_estimators"], md15["exactitud"], marker="o", label="Max Depth = 15")
plt.plot(md20["n_estimators"], md20["exactitud"], marker="o", label="Max Depth = 20")

plt.xticks(n_estimators_valores)
plt.xlabel("n_estimators")
plt.ylabel("Exactitud")
plt.legend()
plt.grid()
plt.show()


#GUARDAR

dump(resultados_df, open("../models/random_forest_n_55_exact_15.sav", "wb"))
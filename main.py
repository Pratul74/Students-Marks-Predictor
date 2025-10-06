import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



data={
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Score":[35,40,50,55,60,65,70,80,85,90],

}

df=pd.DataFrame(data)

# print(df.head())

# plt.scatter(df["Hours_Studied"],df["Score"])
# plt.title("Hours Studied vs Score")
# plt.xlabel("Hours Studied")
# plt.ylabel("Score")
# plt.show()

X=df[["Hours_Studied"]].values
y=df["Score"].values

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train, y_train)
pred=model.predict(X_test)
print("Predictions: ",pred)
print("Actual: ",list(y_test))
print("Predicted Score for 9 hours: ",model.predict([[9]])[0])

score=r2_score(y_test, pred)
print("R2 Accuracy: ",round(score, 2))

plt.scatter(X, y, label="Actual")
plt.plot(X, model.predict(X), color='red', label="Predicted line")
plt.legend()
plt.show()





import pandas as pd

data={
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Score":[35,40,50,55,60,65,70,80,85,90],

}

df=pd.DataFrame(data)

print(df.head())
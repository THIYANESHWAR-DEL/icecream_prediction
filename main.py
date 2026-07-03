import pandas as pd
from model import *
data=pd.read_csv ('icecream_sales.csv')
x=data['Temperature'].values
y=data['Sales'].values
w=0
b=0
alpha=0.001
iterations=1000
w,b,cst=gradient(x,y,w,b,alpha,iterations)
print("Training completed" )
print("Weight:",w)
print("Bias:",b)

while True:
    temp=float(input("\n Enter the temperature to predict ice cream sales (or type 'exit' to quit): "))
    if temp == 'exit':
        break
    else:
        predicted=predict(temp,w,b)
    print(f"predicted ice cream sales at {temp} degree is {predicted:.2f}")





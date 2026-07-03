def predict(x,w,b):
    return w*x+b

def compute_cost(x,y,w,b):
    m=x.shape[0]
    total_cost=0
    for i in range(m):
        predict=w*x[i]+b
        cost=y[i]-predict
        cost=cost**2
        total_cost+=cost
    total_cost/=m
    return total_cost

def compute_gradiant(x,y,w,b):
    m=len(x)
    dj_db=0
    dj_dw=0
    for i in range(m):
        predict=w*x[i]+b
        error=predict-y[i]
        dj_db+=error
        dj_dw+=(error*x[i])
    dj_db=dj_db/m
    dj_dw=dj_dw/m
    return dj_dw,dj_db

def gradient(x,y,w,b,alpha,iterations):
    for i in range(iterations):
        dj_dw,dj_db=compute_gradiant(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
        cost=compute_cost(x,y,w,b)
    return w,b,cost


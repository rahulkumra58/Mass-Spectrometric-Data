import numpy as np
import scipy.spatial.distance as dist
import fileinput 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import*
from sklearn.decomposition import PCA

file = open('Q4-analysis-input.in')
data = file.readlines()
D = pd.to_numeric(data[0]) #dimensions
N = pd.to_numeric(data[1]) #number of rows/patients
dist_metric = pd.to_numeric(data[2]) #number of rows/patients
X = pd.to_numeric(data[3]) #reduced number of dimensions
P = pd.to_numeric(data[4].split()) #reference patient

        

def distance(P, past, distance_metric):
    if distance_metric == 1: #mahattan distance
        distance = dist.cityblock(P,past)
    elif distance_metric == 2: #Euclidean distance
        distance = dist.euclidean(P,past)
    elif distance_metric == 3: # Supremum Distance
        distance = dist.minkowski(P,past,p=float('inf'))
    elif distance_metric == 4: # Cosine Disatnce
        distance = dist.cosine(P,past)
    else:
        distance = 0
    return distance

if(X==1000 or X==100 or X==10 or X==2 or X==1):
    pdata = list(map(int, P))
    pdata = np.array(pdata)
    for i in range(5, N+5):
        all_patients = list(map(int, data[i].split()))
        pdata = np.row_stack((pdata, all_patients))
    pca = PCA(n_components=X, copy=False)
    new_pca = pca.fit_transform(pdata)
    var_list = pca.explained_variance_
    print(pca.explained_variance_)

    l = new_pca.tolist()

    tot_var = 0
    cumulative_var = []

    for i in range(len(var_list)):
        tot_var += var_list[i]
        cumulative_var.append(tot_var)

    plt.figure()
    plt.plot(cumulative_var)
    plt.ylabel('Explained Cumulative Variance')
    plt.xlabel('Number of Principal Components, ' + 'X ='+ str(X))
    plt.title('Cumulative Variance versus Number of Principal Components')
    plt.show()


if(X==-1):
    all_distances=[]
    for i in range(5, N+5):
        all_patients=pd.to_numeric(data[i].split())
        dist = distance(P, all_patients, dist_metric)
        result = (i-4, dist)
        all_distances.append(result)
    sorted_distances = sorted(all_distances, key = lambda x:x[1])
    ans = []
    for dist in sorted_distances:
        ans.append(dist[0])
    a = ans[0:5] 
    for val in a:
    	print(val)
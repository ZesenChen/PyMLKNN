import numpy as np

def knn(train_data, nth_data, k):
    data_num = train_data.shape[0]
    distance = np.zeros((data_num,))    
    Knn = np.zeros((k,))    
    for i in range(data_num):
        sub = train_data[i] - train_data[nth_data]
        distance[i] = (sub**2).sum()
            
    for i in range(k):
        temp = float("inf")
        temp_j = 0
        for j in range(distance.shape[0]):
            if (j != nth_data) and (distance[j] < temp):
                temp = distance[j]
                temp_j = j
        distance[temp_j] = float("inf")
        Knn[i] = temp_j
    return Knn   
            
def knn1(train_data, test_data, k):
    data_num = train_data.shape[0]
    distance =np.zeros((data_num,))
    Knn = np.zeros((k,))
    for i in range(data_num):
        sub = train_data[i] - test_data
        distance[i] = (sub**2).sum()
    
    for i in range(k):
        temp = float("inf")
        temp_j = 0
        for j in range(distance.shape[0]):
            if distance[j] < temp:
                temp = distance[j]
                temp_j = j
        distance[temp_j] = float("inf")
        Knn[i] = temp_j
    return Knn

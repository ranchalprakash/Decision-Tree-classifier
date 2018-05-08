import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
#import pydotplus
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io, sys
from scipy import misc
import pydot

#reading data
df = pd.read_csv('trainX.csv', header = None)

#reading labels
df2 = pd.read_csv('trainY.csv', header = None)


#reading test data
df_test = pd.read_csv('testX.csv', header = None)
df2_test = pd.read_csv('testY.csv', header = None)

#Converting data to matrix
train = df.as_matrix()
label=df2.as_matrix()
test = df_test.as_matrix()
label_test=df2_test.as_matrix()


tree1 = DecisionTreeClassifier(criterion='entropy', min_samples_split=2)

inputvalue = np.arange(0.1, 1, 0.1)
acc_train = []
acc_test = []


#Randomly splitting data and taking 10%, 20%.. for training purposes.

for i in inputvalue:
    x, x1, y, y1 = train_test_split(train, label, train_size=i, test_size=1-i, random_state=10)
    
    tree1 = tree1.fit(x, y)
    train_predict = tree1.predict(x1)
    test_predict = tree1.predict(test)
    
    acc_train.append(accuracy_score(y1, train_predict)*100)
    acc_test.append(accuracy_score(label_test, test_predict)*100)
inputvalue1 = np.append(inputvalue, 1)   
# printing accuracy for 100% training data
tree1 = tree1.fit(train, label)
label_predict = tree1.predict(test)

acc_test.append(accuracy_score(label_test, label_predict)*100)
print(confusion_matrix(label_test, label_predict))
plt.plot(inputvalue1, acc_test)
plt.plot(inputvalue, acc_train)


fig, ax = plt.subplots(nrows=1, ncols=2)
#plt.tight_layout()

plt.subplot(1, 2, 1)
plt.plot(inputvalue1, acc_test)
plt.xlabel("fraction of input",fontsize=17)
plt.ylabel("accuracy(%)",fontsize=17)

plt.subplot(1, 2, 2)
plt.plot(inputvalue, acc_train)
plt.xlabel("fraction of input",fontsize=17)
#plt.ylabel("accuracy(%)", fontsize = 15)

plt.show()

fig.savefig('graph.png') 

n_nodes = tree1.tree_.node_count
children_left = tree1.tree_.children_left
children_right = tree1.tree_.children_right

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True
leaves = np.sum(is_leaves)
print(n_nodes)
print(leaves)
names = ['Radius_m', 'Texture_m', 'Perimeter_m', 'Area_m', 'Smoothness_m', 'Compactness_m', 'Concavity_m', 'Number of concave portions of contour_m', 'Symmetry_m', 'Fractal dimension_m','Radius_v', 'Texture_v', 'Perimeter_v', 'Area_v', 'Smoothness_v', 'Compactness_v', 'Concavity_v', 'Number of concave portions of contour_v', 'Symmetry_v', 'Fractal dimension_v','Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Number of concave portions of contour', 'Symmetry', 'Fractal dimension']



        
        

tree.export_graphviz(tree1, out_file='tree.dot',feature_names=names, class_names=['Beningn','Malign'])

        
        
        
        
        

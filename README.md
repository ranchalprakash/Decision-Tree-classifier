# Decision-Tree-classifier
Classification of Wisconsin Diagnostic Breast Cancer(WDBC) dataset using Decision Tree classifier
Wisconsin Diagnostic Breast Cancer(WDBC) dataset from the UCI repository.
Each row in the dataset represents a sample of biopsied tissue. The tissue for each sample is
imaged and 10 characteristics of the nuclei of cells present in each image are characterized. These
characteristics are: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Number
of concave portions of contour, Symmetry, Fractal dimension. Each sample used in the dataset
is a feature vector of length 30. The first 10 entries in this feature vector are the mean of the characteristics
listed above for each image. The second 10 are the standard deviation and last 10 are the
largest value of each of these characteristics present in each image.

• Training data: ‘ trainX.csv 
consisting of 455 samples, 30 attributes. The label associated
with each sample is provided in ‘ trainY.csv 
. A label of value 1 indicates the sample was
for malignant (cancerous) tissue, 0 indicates the sample was for benign tissue. .
• Test data: ‘ testX.csv 
consisting of 57 samples, 30 attributes. The label associated with
each sample is provided in ‘ testY.csv 

Data is imported using pandas library from csv fle to dataframe . This dataframe is
converteed to numpy array using as _ matrix ().
A decision tree is created using entropy as criterion and The minimum number of samples
required to split an internal node is set to 2.
tree 1 = DecisionTreeClassifer ( criterion =' entropy ', min _ samples _ split =2)
The tree is trained with the training data . The data is randomly split into test and train data
using test _ train split .
'''x , x 1, y , y 1 = train _ test _ split ( train , label , train _ size = i , test _ size =1- i , random _ state =10)
Each set of splitted data is used train diferent tree models . And predictions are done on
both test data and remaining training data .
tree 1 = tree 1. ft ( x , y )
train _ predict = tree 1. predict ( x 1)
test _ predict = tree 1. predict ( test )
The accuracy of the tree on both dataset is calculated .
acc _ train . append ( accuracy _ score ( y 1, train _ predict )*100)
acc _ test . append ( accuracy _ score ( label _ test , test _ predict )*100)
After that similar procedure is done for training the tree with 100% training data .
inputvalue 1 = np . append ( inputvalue , 1)
# printing accuracy for 100% training data
tree 1 = tree 1. ft ( train , label )
label _ predict = tree 1. predict ( test )
acc _ test . append ( accuracy _ score ( label _ test , label _ predict )*100)
print ( confusion _ matrix ( label _ test , label _ predict ))
The accuracy vs % of training data is plotted and the graph is stored in graph . png .
plt . subplot (1, 2, 1)
plt . plot ( inputvalue 1, acc _ test )
plt . xlabel (" fraction of input ")plt . ylabel (" accuracy (%)")
plt . subplot (1, 2, 2)
plt . plot ( inputvalue , acc _ train )
plt . xlabel (" fraction of input ")
# plt . ylabel (" accuracy (%)", fontsize = 15)
plt . show ()
fg . savefg (' graph . png ')
Graph'''

Figure 1 plots the accuracy score of prediction for test data when 0.2, 0.4, 0.6.. fractions of
training data is taken for training .
Figure 2 plots the accuracy score of prediction for the remaining training data when 0.2, 0.4,
0.6... of training data is taken for training .
From the graph we can see that training the tree with 30% data gives the best result .
The Confusion matrix was obtained to be :
31 1
3 22
The decision tree is converted to dot . fle using ,
'''tree . export _ graphviz ( tree 1, out _ fle =' tree . dot ', feature _ names = names ,class _ names =[' Beningn ',' Malign '])'''
The dot fle is converted into pdf using
'dot - Tpdf tree . dot - o tree . pdf'
The diagram of desicion tree is attached below .
Total number of nodes is 25
Total number of leaf nodes is 14
.

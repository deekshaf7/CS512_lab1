# %%
#Github code - https://github.com/deekshaf7/CS512_lab1.git
#Required Libraries and packages
!pip3 install -U liblinear-official
from liblinear.liblinearutil import *
from utils import *

# %%
#Q3(a)Accuracy on letter-wise prediction -the percentage of correctly predicted letters on the whole test set 

c_values = [1, 10, 100, 1000, 5000, 10000, 15000, 20000]

# %%
#1. Accuracy on letter-wise prediction using CRF
#Calculation of accuracy
nc = len(c_values)
accuracy_values_crf = [0] * nc
#Plot the accuracy vs c values for CRF
model = "CRF"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, accuracy_values_crf, model, x_label,3,0)

# %%
#2. Accuracy on letter-wise prediction using SVM-Struct(svm-hmm)
#Follow requirement.txt and make file to install the required packages
#Calculation of accuracy
accuracy_values_svm_struct = {}
#It will take 2-3 minutes to run
for c in c_values:
    accuracy_values_svm_struct[c] = svm_struct(c)

accuracy_values_array_struct = list(accuracy_values_svm_struct.values())

#Plot the accuracy vs c values for SVM-Struct
model = "SVM-Struct"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, accuracy_values_array_struct, model, x_label,3,0)

# %%
#3. Accuracy on letter-wise prediction using SVM-MC 
#Calculation of accuracy
accuracy_values_svm_mc = {}
for c in c_values:
    accuracy_values_svm_mc[c] = svm_mc(c,0,0)

accuracy_values_array_mc = list(accuracy_values_svm_mc.values())

#Plot the graph
model = "SVM-MC"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, accuracy_values_array_mc, model, x_label,3,0)

# %%
#Q3(b)Accuracy on word-wise prediction accuracy on test data
#1. Accuracy on word-wise prediction using CRF
#Calculation of accuracy
nc = len(c_values)
w_accuracy_values_crf = [0] * nc
#Plot the accuracy vs c values for CRF
model = "CRF"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, w_accuracy_values_crf, model, x_label,3,1)
#---------------------------------------------------------------------------

# %%
#2. Accuracy on word-wise prediction using SVM-Struct(svm-hmm)
#Follow requirement.txt and make file to install the required packages
#Calculation of accuracy
w_accuracy_values_svm_struct = {}
#It will take 2-3 minutes to run
for c in c_values:
    w_accuracy_values_svm_struct[c] = svm_struct_w(c)

w_accuracy_values_array_struct = list(w_accuracy_values_svm_struct.values())

#Plot the accuracy vs c values for SVM-Struct
model = "SVM-Struct"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, accuracy_values_array_struct, model, x_label,3,1)
#---------------------------------------------------------------------------

# %%
#3. Accuracy on word-wise prediction using SVM-MC 
#Calculation of accuracy
w_accuracy_values_svm_mc = {}
for c in c_values:
    w_accuracy_values_svm_mc[c] = svm_mc_w(c,0,0)

w_accuracy_values_array_mc = list(w_accuracy_values_svm_mc.values())

#Plot the graph
model = "SVM-MC"  # Name of the model
x_label = "C"  # Description of what parameter is varied
plot_accuracy(c_values, w_accuracy_values_array_mc, model, x_label,3,1)

# %%
#Q5 (a) Accuracy on letter-wise prediction -the percentage of correctly predicted letters on the whole test set

#plot the letter-wise prediction accuracy on test data vs x lines of transformations on training data
x_values = [0, 500, 1000, 1500, 2000]

#data transformation
for x in x_values:
    transform_training_data_rotate_translate(x)

#1 CRF part
# Best value of c for CRF found in 3a is <->
#call the crf function to get the accuracy values
accuracy_values = [0.82, 0.84, 0.85, 0.86, 0.87]  # Example accuracy values for each dataset size

model = "CRF"  # Name of the model
x_label = "x"  # Description of what parameter is varied ]

plot_accuracy(x_values, accuracy_values, model, x_label,5,0)

# %%
#2 SVM-MC part
# Best value of c for CRF found in 3a is 1000
c = 1000
accuracy_values_svm_mc_x = {}
#Preprocess the data to work with the SVM-MC model and Calculate accuracy values for each x value
for x in x_values:
    accuracy_values_svm_mc_x[x] = svm_mc(c, x,1)
    
print(accuracy_values_svm_mc_x)
accuracy_values_array_mc_x = list(accuracy_values_svm_mc_x.values())

plot_accuracy(x_values, accuracy_values_array_mc_x, "SVM-MC", "x",5,0)

#Observations


# %%
#Q5 (b) Accuracy on letter-wise prediction -the percentage of correctly predicted letters on the whole test set




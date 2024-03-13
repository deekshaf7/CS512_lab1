import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
import cv2
from liblinear.liblinearutil import *
import subprocess
import sys

def transform_training_data_rotate_translate(x):
    file_path1 = 'transformed_training_data/train_struct_' + str(x) + '.txt'
    file_path2 = 'transformed_training_data/train_struct_transformed' + str(x) + '.txt'
    file_paths = [file_path1, file_path2]

    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        # Check if the directory exists, if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Check if the file exists, if so, delete it
        if os.path.exists(file_path):
            os.remove(file_path)

    file = open('data/transform.txt', 'r')
    lines = file.readlines()
    word_id_r = []
    shift_x = []
    word_id_t = []
    offset_t = []
    for line in lines[:x]: # x
        lines = line.split()
        if lines[0] == 'r':
            word_id_r.append(lines[1])
            shift_x.append(lines[2])
        else:
            word_id_t.append(lines[1])
            offset_t.append([lines[2], lines[3]])

    df = pd.read_csv("data/train.txt", header=None, sep=' ')
    letters = df[1]
    orig_word_id = df[3]
    pixels = df.iloc[:,5:133]
    pixels = pixels.reset_index(drop=True)
    pixels = np.array(pixels)

    word_letters = {}
    word_list = {}
    for i in range(len(orig_word_id)):
        if orig_word_id[i] not in word_list.keys():
            word_list[orig_word_id[i]] = [pixels[i]]
            word_letters[orig_word_id[i]] = [letters[i]]
        else:
            word_list[orig_word_id[i]].append(pixels[i])
            word_letters[orig_word_id[i]].append(letters[i])

    # Open file in write mode to overwrite
    with open('transformed_training_data/train_struct_'+str(x)+'.txt', 'w') as file:
        for i in word_id_r:
            pixels = word_list[int(i)]
            letters = word_letters[int(i)]
            shift = shift_x[word_id_r.index(i)]
            for j in range(len(pixels)):
                pixel = pixels[j].reshape((16, 8))
                pixel = pixel.astype(np.uint8)

                image_center = tuple(np.array(pixel.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, int(shift), 1.0)
                result = cv2.warpAffine(pixel, rot_mat, pixel.shape[1::-1], flags=cv2.INTER_LINEAR)

                indices = np.where(result.reshape(128) != 0)[0]

                data = []
                for k in indices:
                    data.append(str(k)+':1')

                file.write(str(ord(letters[j])- ord('a'))+' qid:'+i+' '+' '.join(data) + '\n')

        for i in word_id_t:
            pixels = word_list[int(i)]
            letters = word_letters[int(i)]
            offset = offset_t[word_id_t.index(i)]
            for j in range(len(pixels)):
                pixel = pixels[j].reshape((16, 8))
                pixel = pixel.astype(np.uint8)

                result = cv2.warpAffine(pixel, np.float32([[1, 0, int(offset[0])], [0, 1, int(offset[1])]]), (8, 16))

                indices = np.where(result.reshape(128) != 0)[0]

                data = []
                for k in indices:
                    data.append(str(k)+':1')

                file.write(str(ord(letters[j])- ord('a'))+' qid:'+i+' '+' '.join(data) + '\n')
    file.close()

    with open('transformed_training_data/train_struct_'+str(x)+'.txt', 'r') as file:
        #read lines from the file        
        input_lines = file.readlines()

        # Group lines by qid in the input file
        qid_to_lines = {}
        for line in input_lines:
            parts = line.split()
            if len(parts) > 1:
                qid = parts[1]  # Assuming the qid is the second word
                if qid not in qid_to_lines:
                    qid_to_lines[qid] = []
                qid_to_lines[qid].append(line)
        #Read reference file lines
        with open('data/train_struct.txt', 'r') as file:
            reference_lines = file.readlines()
        # Replace lines in reference file based on qid match
        for i, line in enumerate(reference_lines):
            words = line.split()
            if len(words) > 1 and words[1] in qid_to_lines and qid_to_lines[words[1]]:
                # Replace line with the first line associated with this qid, then remove it from the list
                reference_lines[i] = qid_to_lines[words[1]].pop(0)
                if not qid_to_lines[words[1]]:  # If no more lines left for this qid, delete it from the dict
                    del qid_to_lines[words[1]]
        # Save the modified content to the output file 
        with open('transformed_training_data/train_struct_transformed'+str(x)+'.txt', 'w') as file:
            file.writelines(reference_lines)    
    file.close()

def plot_accuracy(x_values, accuracy_values, model, x_label,que,word):
    """
    Plot the prediction accuracy vs. x_values with a title based on the model name and save the plot in the 'results' folder,
    with a custom label for the x-axis.

    Parameters:
    - x_values: List of integers or floats representing the x values (e.g., dataset sizes, parameter values).
    - accuracy_values: List of floats representing the prediction accuracy for each x value.
    - model: String representing the name of the model, used to construct the plot title.
    - x_label: String representing the custom label for the x-axis.
    """
    title = f"Prediction Accuracy vs {x_label} value for {model}"
    plt.figure(figsize=(10, 6))  # Create a new figure with a specific size
    plt.plot(x_values, accuracy_values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(f'{x_label}---------->')
    if word == 0:
        plt.ylabel('Letter-wise Prediction Accuracy (%)------------->')
    else:   
        plt.ylabel('Word-wise Prediction Accuracy (%)------------->')
    if que == 3:
        plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xticks(x_values, labels=[str(x) for x in x_values])
    if que == 3:
        plt.xscale('log')

    # Ensure the 'Results' directory exists
    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)

    # Save the plot in the 'Results' folder
    file_name = os.path.join(results_dir, f'{model}_Accuracy_vs_{x_label}.png')
    plt.savefig(file_name)
    plt.show()  

def svm_mc(c, x,t):
    """
    Trains an SVM model with a given C parameter adjusted by the number of training examples,
    and evaluates its accuracy on a test set.

    Parameters:
    - c: The base C value for SVM training.
    - training_data_path: Path to the file containing the training data in LIBSVM format.
    - testing_data_path: Path to the file containing the testing data in LIBSVM format.

    Returns:
    - A float representing the prediction accuracy of the model on the test set.
    """
    # Prepare the training data set 
    testing_data_path = 'data/test_struct.txt'
    svm_mc_data_dir = 'svm_mc_data'
    os.makedirs(svm_mc_data_dir, exist_ok=True)

    if t == 0:
        training_data_path = 'data/train_struct.txt'
    else:
        training_data_path = 'transformed_training_data/train_struct_transformed' + str(x) + '.txt'
    
    # Preprocess the training data
    training_data_path_p = transform_training_data(training_data_path,x)
    testing_data_path_p = transform_test_data(testing_data_path)
   
    # Read the training and testing data
    y_train, x_train = svm_read_problem(training_data_path_p)
    y_test, x_test = svm_read_problem(testing_data_path_p)

    # Adjust the C value based on the number of training examples
    number_of_training_examples = len(y_train)
    n = c / number_of_training_examples

    # Train the model
    model = train(y_train, x_train, f'-c {n}')

    # Predict and evaluate accuracy
    p_label, p_acc, p_val = predict(y_test, x_test, model)
    
    # p_acc is a tuple containing accuracy, MSE, and SCC. We return only the accuracy.
    accuracy = p_acc[0]
    return accuracy

def transform_training_data(input_file_path, x):
    output_file_path = 'svm_mc_data/train_struct_transformed' + str(x) + '.txt'

    with open(input_file_path, 'r') as file:
        modified_lines = []
        for line in file:
            parts = line.split()
            if len(parts) > 1:
                del parts[1]
            modified_line = ' '.join(parts)
            modified_lines.append(modified_line)

    with open(output_file_path, 'w') as output_file:
        for modified_line in modified_lines:
            output_file.write(modified_line + '\n')

    return output_file_path

def transform_test_data(input_file_path):
    output_file_path = 'svm_mc_data/test_struct.txt'
    
    with open(input_file_path, 'r') as file:
        modified_lines = []
        for line in file:
            parts = line.split()
            if len(parts) > 1:
                del parts[1]
            modified_line = ' '.join(parts)
            modified_lines.append(modified_line)

    with open(output_file_path, 'w') as output_file:
        for modified_line in modified_lines:
            output_file.write(modified_line + '\n')

    return output_file_path

def svm_struct(c_value):
    command1 = f"svm_hmm/svm_hmm_learn -c {c_value} data/train_struct.txt modelfile.dat"
    command2 = "svm_hmm/svm_hmm_classify data/test_struct.txt modelfile.dat classify.tags"

    # Execute the training command
    process1 = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process1.returncode != 0:
        return "Error during training:", process1.stderr

    # Execute the classification command and capture its output
    process2 = subprocess.run(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process2.returncode != 0:
        return "Error during classification:", process2.stderr

    output = process2.stdout

    zero_one_error_match = re.search(r"Zero/one-error on test set: (\d+\.\d+)%", output)

    if zero_one_error_match:
        zero_one_error_value = zero_one_error_match.group(1)
        accuracy = 100 - float(zero_one_error_value)
        return accuracy
    else:
        return "Zero/one-error value not found."

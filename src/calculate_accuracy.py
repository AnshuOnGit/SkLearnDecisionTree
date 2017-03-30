import numpy as np
from sklearn import tree
import csv

filepath = "/Users/anskumar/repos/hackathon/SkLearnDecisionTree/resources/EXAMPLE_DATASET_HACKATHON2017_CSV.csv"
total_rows_count = 0


clf = tree.DecisionTreeClassifier()
with open(filepath, 'r') as file:
    total_rows_count = len(list(csv.reader(file, delimiter=',')))

training_rows_count  = int(total_rows_count*0.8)
test_rows_count = total_rows_count - training_rows_count

training_rows= np.genfromtxt(filepath
                         ,delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), max_rows=training_rows_count, skip_header=1)

test_rows =  np.genfromtxt(filepath
                         ,delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), max_rows=test_rows_count, skip_header=training_rows_count)


for output in range(9, 12):
    success = 0
    print("calculating accuracy for %d row" % output)
    output1 = np.genfromtxt(filepath
                            , delimiter=',', usecols=(output), max_rows=training_rows_count, skip_header=1, dtype=str)

    expected_outputs = np.genfromtxt(filepath
                                     , delimiter=',', usecols=(output), max_rows=test_rows_count,
                                     skip_header=training_rows_count, dtype=str)

    clf.fit(training_rows, output1)
    for i in range(0, test_rows_count):
        test_row = test_rows[i : i + 1 ]
        predicted_output = clf.predict(test_row)
        if(predicted_output[0] == expected_outputs[i]):
            success += 1
        #feedback is reducing accuracy
        #clf.fit(np.concatenate((training_rows, test_row), axis=0), np.concatenate((output1, predicted_output), axis=0))

    print("total = %d" % (test_rows_count))
    print("success = %d" % (success))
    print("accuracy = %d percentage" % (success/test_rows_count * 100) )



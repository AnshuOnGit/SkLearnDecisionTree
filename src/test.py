import numpy as np
from sklearn import tree
from scipy.sparse import csr_matrix
import pandas as pd

clf = tree.DecisionTreeClassifier()

train_df = pd.read_csv("/home/anshu/smart_india_hackathon/dev_env/resources/EXAMPLE_DATASET_HACKATHON2017_CSV.csv")

clf.fit(train_df[['N1', 'N2', 'N3', 'N4', 'N5','N6', 'N7', 'N8','N9']].values, train_df['M1'])

print(clf.predict([[0.0761,0.0491,9.6023752203,1.1548798387,10.1435503406,1.4464270979,28.8001,21.6001,0.8861463886]]))

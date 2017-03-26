import numpy as np
from sklearn import tree
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz



clf = tree.DecisionTreeClassifier()

train_df = pd.read_csv("/home/deepa_panicker/SkLearnDecisionTree/resources/EXAMPLE_DATASET_HACKATHON2017_CSV.csv")

clf.fit(train_df[['N1', 'N2', 'N3', 'N4', 'N5','N6', 'N7', 'N8','N9']].values, train_df['M3 '])

# print(clf.predict([[0.0761,0.0491,9.6023752203,1.1548798387,10.1435503406,1.4464270979,28.8001,21.6001,0.8861463886]]))


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    from subprocess import check_call
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


visualize_tree(clf, ['N1',' N2', 'N3', 'N4', 'N5','N6', 'N7', 'N8','N9'])


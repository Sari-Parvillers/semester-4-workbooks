"""
The ID3 algorithm for building a decision tree. 

Modified by Kunhe, at 09. Mar. 2024.

ID3 only supports categorical data! 

This function recursively constructs a decision tree by selecting the feature
that maximizes information gain at each step. It splits the dataset on the best
feature, then recursively applies the same process to the resulting subsets.

Function names:
entropy, InfoGain, ID3, predictDataset, print_tree

Parameters:
- data: The dataset with both X and Y.
- originaldata: When called outside, put the same dataset here. This parameter is for recursion.
- features: The list of feature names that are still considered for splitting. The column name of true classes should not be included.
- target_attribute_name: The name of the target attribute (class label).

Returns:
- A decision tree represented as a nested dictionary. Leaves are class labels.

Source:
- Adapted from https://github.com/mantis522/Daily_python/blob/0c509d1a5d5c4dcdb21ccfeb53055bfb42f41848/Machine_learning/decision_tree.py
"""

import numpy as np


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i] /
                                                          np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(
        data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
        
def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):
    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name]) \
            [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.
    elif len(features) == 0:
        return parent_node_class

    # If none of the above holds true, grow the tree!
    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]

        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, data, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        return (tree)
        
def predict_single(query, tree, default="No data"):
    # Check if tree is a string
    if isinstance(tree, dict):   
        for key in tree.keys():
            if key in list(query.keys()):
                value = query[key]
                if value in tree[key].keys():
                    return predict_single(query, tree[key][value], default)
                else:
                    return default
            else:
                raise ValueError('Attribute not found.')
    else:     
        return tree

def predictDataset(test_data, tree, default="No data"):
    # Create an empty list to store predictions
    predictions = []
    
    # For each row in the test data, predict the outcome
    for i in range(len(test_data)):
        predictions.append(predict_single(test_data.iloc[i], tree, default))
    
    return predictions

def print_tree(tree, depth=0, prefix=""):
    if isinstance(tree, dict):
        # Iterate through the attributes and their values in the tree
        for idx, (attribute, branches) in enumerate(tree.items()):
            # First attribute does not get the "|---" prefix
            if depth > 0 or idx > 0:
                attr_prefix = prefix + "|--- "
            else:
                attr_prefix = prefix
            
            # Print the attribute name
            print(f"{attr_prefix}{attribute}")
            for value, subtree in branches.items():
                # Construct the new prefix for the subtree
                # If this is the first attribute or value, don't add the "|   " part yet
                if depth > 0 or idx > 0:
                    new_prefix = prefix + "|   "
                else:
                    new_prefix = prefix
                
                # Print the attribute value with the "|---" prefix
                print(f"{new_prefix}|--- {value}")
                # Recursive call to print the subtree
                print_tree(subtree, depth + 1, new_prefix + "|   ")
    else:
        # If we reach a leaf, print the class label with the "|---" prefix
        print(f"{prefix}|--- {tree}")
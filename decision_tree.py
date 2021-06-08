import csv
from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

 
      
class DecisionTree(object):
    
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = 20
        self.depth = 0
        self.classified=0
        
        #pass

    def learn(self, X, y):
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        largest_info_gain = 0
        best_criteria = {}    # Feature index and threshold
        best_sets = None        # Subsets of the data
        info_gain=0
        
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        n_samples, n_features = np.shape(X)
        p={}

        for feature_i in range(n_features):

            X=np.array(X)
            unique_values = np.unique(X[:,feature_i],axis=0)
            example_value = unique_values[0]
            
            #handling of the integer and float features
            if isinstance(example_value, np.float64) or isinstance(example_value, np.int):
                
                #using median of the feature for splitting the dataset.
                split_val=np.median(X[:,feature_i])
                X_left, X_right, y_left, y_right = partition_classes(X,y, feature_i, split_val)
                left_right=list([y_left,y_right])

                if len(X_left) > 0 and len(X_right) > 0:                        
                    info_gain = information_gain(y, left_right)
                    best_criteria = {"feature_i": feature_i, "split_val": split_val}
                    best_sets = {
                        "leftX": X_left,   # X of left subtree
                        "lefty": y_left,   # y of left subtree
                        "rightX": X_right,  # X of right subtree
                        "righty": y_right  # y of right subtree
                                }
                    largest_info_gain = info_gain
                
            else:
                try:
                    #handling of the integer and float features in the form of strings
                    if (ast.literal_eval(example_value)):
                        xxp = []
                        for i in range(len(X[:,feature_i])):
                            temp0=ast.literal_eval(X[:,feature_i][i])
                            xxp= np.append(xxp, temp0)
                        
                        split_val=np.median(xxp)
                            #split_val=pp/len(X[:,feature_i])#np.mean(X[:,feature_i].)
                            
                        X_left, X_right, y_left, y_right = partition_classes(X,y, feature_i, split_val)
                        left_right=list([y_left,y_right])
                        if len(X_left) > 0 and len(X_right) > 0:
                            info_gain = information_gain(y, left_right)
                            best_criteria = {"feature_i": feature_i, "split_val": split_val}
                            best_sets = {
                                        "leftX": X_left,   # X of left subtree
                                        "lefty": y_left,   # y of left subtree
                                        "rightX": X_right,  # X of right subtree
                                        "righty": y_right  # y of right subtree
                                        }
                            largest_info_gain = info_gain
                    
                    
                except ValueError: 
                    #handling of categorical features
                    for threshold in unique_values:
                        #splitting dataset on each value of the categorical variable
                        X_left, X_right, y_left, y_right = partition_classes(X,y, feature_i, threshold)
                        left_right=list([y_left,y_right])
                    
                        if len(X_left) > 0 and len(X_right) > 0:
                            info_gain = information_gain(y, left_right)
                    
                        if info_gain > largest_info_gain:
                            #capturing the feature with the largest information gain.
                            largest_info_gain = info_gain
                            best_criteria = {"feature_i": feature_i, "split_val": threshold}
                            best_sets = {
                                        "leftX": X_left,   # X of left subtree
                                        "lefty": y_left,   # y of left subtree
                                        "rightX": X_right,  # X of right subtree
                                        "righty": y_right  # y of right subtree
                                }
                        
        largest_info_gain=largest_info_gain
   
        if largest_info_gain > 0:
            self.depth += 1
            temp=self.depth

        if n_samples >= 2  and largest_info_gain > 0 and self.depth <= self.max_depth:
            p['feature_id'] = best_criteria['feature_i']
            p['split_val'] = best_criteria['split_val']
            p['depth'] = temp
            
            
            p['left']=self.learn(best_sets['leftX'],best_sets['lefty'])
            p['right']=self.learn(best_sets['rightX'],best_sets['righty'])
                

                
            
            self.tree=p
            return p
     
        else:
            #DecisionTree.length=1
            return y

        
    def classify(self, record):
        #parent=None
      
        record=np.array(record)
        temp_tree=self.tree
        
        #recursive function for traversing through the tree
        def tree_traverse(node):

            if (isinstance(node, dict)):
                #print(self.tree)

                if isinstance(node['split_val'],str):
                    if record[node['feature_id']] == node['split_val']:
                        next_node=node['left']
                    else:
                        next_node=node['right']
                else:
                    try:

                        if ast.literal_eval(record[node['feature_id']]) <= node['split_val']:
                            next_node=node['left']
                        else:
                            next_node=node['right']   
                    except ValueError: 
                        if record[node['feature_id']] <= node['split_val']:
                            next_node=node['left']
                        else:
                            next_node=node['right']  
             
                       
                tree_traverse(next_node)
            else:

                self.classified=node

                
        tree_traverse(temp_tree)
        px=self.classified.astype(int)
        px=np.ravel(px)
        counts = np.bincount(px)  
        
        return np.argmax(counts)  
                

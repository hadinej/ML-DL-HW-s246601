from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        
        self.root = root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
            
        if self.split == 'train': 
            #first reading train.txt and assigning indexes
            file_train = open(self.root+'train.txt',"r")
            self.read_train_lines=file_train.readlines()

            # ------------------------------------------------------------------------
            # Python code to initialize a dictionary 
            # with only keys from a list  # List of keys 
            dict_keys=[]
            for i in range(len(self.read_train_lines)):
                dict_keys.append(self.read_train_lines[i][0:12])
            dict_keys = list(dict.fromkeys(dict_keys))
            
            # make np array to be used later for __getitem__
            dict_keys_arr=np.array(dict_keys)
            self.dict_keys_arr=np.delete(dict_keys_arr,4)

            # using fromkeys() method 
            index_dict = dict.fromkeys(dict_keys, []) 

            # now assigning each index to its class hence its dict key
            k=0
            dict_val=[]
            for j in range(len(dict_keys)):
                dict_val=[]
                while k<len(self.read_train_lines) and self.read_train_lines[k][0:12] == dict_keys[j]:
                    dict_val.append(k)
                    k+=1
                index_dict[str(dict_keys[j])]=dict_val
            # -------------------------------------------------------------------------------------------  
            # now removing BACKGROUND_Google by removing it from dict.
            del index_dict['BACKGROUND_G']
            
            self.index_dict=index_dict
            
            

        if self.split == 'test': 
            #first reading train.txt and assigning indexes
            file_test = open(self.root+'test.txt',"r")
            self.read_test_lines=file_test.readlines()

            # ------------------------------------------------------------------------
            # Python code to initialize a dictionary 
            # with only keys from a list  # List of keys 
            dict_keys=[]
            for i in range(len(self.read_test_lines)):
                dict_keys.append(self.read_test_lines[i][0:12])
            dict_keys = list(dict.fromkeys(dict_keys))
            
            # make np array to be used later for __getitem__
            dict_keys_arr=np.array(dict_keys)
            self.dict_keys_arr=np.delete(dict_keys_arr,4)

            # using fromkeys() method 
            index_dict = dict.fromkeys(dict_keys, []) 

            # now assigning each index to its class hence its dict key
            k=0
            dict_val=[]
            for j in range(len(dict_keys)):
                dict_val=[]
                while k<len(self.read_test_lines) and self.read_test_lines[k][0:12] == dict_keys[j]:
                    dict_val.append(k)
                    k+=1
                index_dict[str(dict_keys[j])]=dict_val
            # -------------------------------------------------------------------------------------------   
            # now removing BACKGROUND_Google by removing it from dict.
            del index_dict['BACKGROUND_G']
            
            
            self.index_dict=index_dict
    # ---------------------------------------------------------------------------------------------------
   
    def selectindex(self,type):
        # now splitting the indexes for train and validation
        #first convert to pandas
        index_df=pd.Series(self.index_dict)
        print(index_df)

        index_df_train=pd.Series(self.index_dict)
        index_df_val=pd.Series(self.index_dict)
        index_df_train.loc[:] = np.nan
        index_df_val.loc[:] = np.nan

        #spliting each dataframe to two dataframes one for test and one for validation
        #each data frame index is the class of Images
        for i in range(len(index_df.index)):
            index_train, index_val = train_test_split(index_df.iloc[i], test_size=0.5, random_state=42)
            index_df_train.iloc[i]=index_train
            index_df_val.iloc[i]=index_val
        
        flattened_list_train_index = [y for x in index_df_train.values for y in x]
        flattened_list_val_index = [y for x in index_df_val.values for y in x]
        
        if type == 'train':
            return flattened_list_train_index
        if type == 'validation':
            return flattened_list_val_index
        
        '''
        this creats the split indexes for test and validation with 50-50 shares
        '''
            
                
         # ----------------------------------------------------------------------------------------------
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args: 
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        
        # -----------------------------------------------------------------------------------------------
        #creating a function for retreving each index key which is the label and converting to int
        def get_key(val):

            for n in range(len(list(self.index_dict.values()))):
                if val in list(self.index_dict.values())[n]:
                    result = np.where(self.dict_keys_arr == list(self.index_dict.keys())[n])
                    return int(result[0][0])
                
#         def get_key(val):
#             for n in range(len(list(self.index_dict.values()))):
#                 if val in list(self.index_dict.values())[n]:
#                     return list(self.index_dict.keys())[n]


                
#         def to_categorical(y, num_classes):
#             """ 1-hot encodes a tensor """
#             return np.eye(num_classes, dtype='uint8')[y]


        if self.split == 'train':
            readlines=self.read_train_lines
        if self.split == 'test':
            readlines=self.read_test_lines
        
        
        image, label = (pil_loader(self.root+'101_ObjectCategories/'+readlines[index][:-1]),get_key(index)) 

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    
    
    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        if self.split == 'train':
            length = len(self.read_train_lines) # Provide a way to get the length (number of elements) of the dataset
            return length
        if self.split == 'test':
            length = len(self.read_test_lines) # Provide a way to get the length (number of elements) of the dataset
            return length
#import necessary packages
from sklearn import preprocessing
import pandas as pd

def label_one_hot_encode(df, col_name):
    """First label encode the specified column and then one hot encode it
    Arguments:
        df: pandas.DataFrame that consists of the column to be encoded
        col_name: categorical attribute to be encoded
    Returns:
        label_encoder, one_hot_encoder and transformed column in form of pandas.Series
    """
    #Label encode the feature (text categorical attribute to numerical)
    lbl_encoder = preprocessing.LabelEncoder()
    labels = lbl_encoder.fit_transform(df[col_name])
    df[col_name + '_label'] = labels
    #One hot encode the lablelled columns
    ohe = preprocessing.OneHotEncoder()
    feature_ohe = ohe.fit_transform(df[[col_name + '_label']]).toarray()
    #Construct the column names for one hot encoded features
    feature_labels = [col_name+'_'+str(class_label) for class_label in lbl_encoder.classes_]
    df_features = pd.DataFrame(feature_ohe, columns=feature_labels)
    return lbl_encoder, ohe, df_features
    
"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from collections import defaultdict #diccionarios con valor por defecto
from scipy.sparse import hstack

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

def label_encoder(s):
    if s=='new':
        return 1
    else:
        return 0

if __name__ == "__main__":
    print("Loading dataset...")
    # Train and test data following sklearn naming conventions
    # X_train (X_test too) is a list of dicts with information about each item.
    # y_train (y_test too) contains the labels to be predicted (new or used).
    # The label of X_train[i] is y_train[i].
    # The label of X_test[i] is y_test[i].
    X_train, y_train, X_test, y_test = build_dataset()

    # Insert your code below this line:
    
    print("Preprocessing the data...")
    tabla_train=pd.DataFrame.from_dict(X_train, orient='columns') #las keys son las columnas

    tabla_train=tabla_train[['title','price','listing_type_id','initial_quantity','sold_quantity','category_id','condition']]

    tabla_test = pd.DataFrame.from_dict(X_test, orient='columns')[['title','price','listing_type_id','initial_quantity','sold_quantity','category_id']]

    y_train_encoded=np.array(list(map(label_encoder,y_train)))

    y_test_encoded=np.array(list(map(label_encoder,y_test)))    
    
    



    #texto
    sw=stopwords.words('spanish')
    sw.remove('sin')
    sw.remove('con')

    vectorizador_bow=CountVectorizer(stop_words=sw)
    x_train_bow=vectorizador_bow.fit_transform(tabla_train['title'])

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_bow)

    x_test_bow = vectorizador_bow.transform(tabla_test['title'])
    x_test_tfidf = tfidf_transformer.transform(x_test_bow)
    
    
    
    
    #features
 #sumamos uno para que los codigos empiecen en 1 y no en 0
    cat_codes=tabla_train['category_id'].astype('category').cat.codes + 1
    list_codes=tabla_train['listing_type_id'].astype('category').cat.codes + 1

    #encodeamos con un diccionario, defaultdic es para que las claves que no estén tengan un valor por defecto (el 0)
    encoder_cat = {key:value for (key,value) in np.array([tabla_train['category_id'],cat_codes]).T}
    encoder_cat=defaultdict(int ,encoder_cat)
    decoder_cat = {value:key for (key,value) in encoder_cat.items()}
    decoder_cat=defaultdict(int ,decoder_cat)

    encoder_type = {key:value for (key,value) in np.array([tabla_train['listing_type_id'],cat_codes]).T}
    encoder_type=defaultdict(int ,encoder_type)
    decoder_type = {value:key for (key,value) in encoder_cat.items()}
    decoder_type=defaultdict(int ,decoder_type)


    #encodeamos las categoricas
    x_train_sc = (
            np.array([tabla_train['category_id'].map(encoder_cat),
                      tabla_train['listing_type_id'].map(encoder_type)])
            ).T

    #y las unimos con las otras variables
    x_train_features = pd.concat([pd.DataFrame(x_train_sc),tabla_train['initial_quantity'], tabla_train['sold_quantity']],axis=1).to_numpy()


    x_test_sc = (np.array([tabla_test['category_id'].map(encoder_cat),
                      tabla_test['listing_type_id'].map(encoder_type)]).T
            )
    x_test_features = pd.concat([pd.DataFrame(x_test_sc),tabla_test['initial_quantity'], tabla_test['sold_quantity']],axis=1).to_numpy()


    # las matrices de tf-idf son esparas, se concatenan con la funcion hstack
    x_train = hstack([x_train_tfidf,x_train_features])
    x_test = hstack([x_test_tfidf,x_test_features])


	#entrenar
    print("Training model...")
    tree = DecisionTreeClassifier(max_depth = 40).fit(x_train, y_train_encoded)
    
    
    print("Evaluation")
    ####evaluar
    y_pred=tree.predict(x_test)
    y_pred_prob=tree.predict_proba(x_test)
    
    ##accuracy
    acc_tree= (y_pred == y_test_encoded).sum()/len(y_test)
    print('Accuracy:',round(acc_tree,3))
    

    #roc score con positive=new, hay que darle probabilidades, no las predicciones
    auc=roc_auc_score(y_true=y_test_encoded, y_score=y_pred_prob[:,1],labels=[1,0])
    print('AUC_ROC-SCORE:', round(auc,3))	

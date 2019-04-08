import os
import sys
import json
import pandas
import numpy
import keras
import keras.models
from keras.models import Sequential,Model
import keras.layers
from keras.layers import *
import sklearn.preprocessing
import sklearn.metrics
import sklearn.tree


def basic_conv_model(features,targets):
    features = features.copy()
    targets = targets.copy()
    
    cat_columns = ['AREA', 'PROBLEM', 'TOPIC']
    for colname in cat_columns:
        prefix = colname
        factor_features = pandas.get_dummies(features[colname])
        factor_features.columns = [prefix+'_'+x for x in factor_features.columns]
        features = pandas.concat([features.drop(colname,axis=1),factor_features],axis=1)
    
    print(features.columns)
    features = features.drop('STUDENT_ID',axis=1)
    
    class_weights = {0: 1., 
                     1: 2.}
    
    #features = features.fillna(0.0)
    # attempt to do inputation on top of this instead
    feature_cols = features.columns.tolist()
    indexes = features.index.tolist()
    features = sklearn.preprocessing.Imputer().fit_transform(features)
    
    features = pandas.DataFrame(features,index=indexes,columns=feature_cols)
    
    # Attempt random fill values for preprocessing
    M = len(features.index)
    N = len(features.columns)
    ran = pandas.DataFrame(numpy.random.randn(M,N)*0.1, columns=features.columns, index=features.index)
    features.update(ran,overwrite=False)
    
    input_shape = (features.shape[1],)
    batch_size = 32
    model = Sequential()
    model.add(Dense(2**8,input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(512)) #, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))


    optimizer = keras.optimizers.Adadelta(lr=0.75, rho=0.95, epsilon=0.001, decay=1.)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    model.fit(features, targets, batch_size=batch_size, 
              epochs=10,class_weight=class_weights)
    score = model.evaluate(features, targets)
    predictions = model.predict(features)
    
    print(score)
    print(predictions)
    print(sklearn.metrics.f1_score(targets,pandas.Series(predictions.flatten()).map(int)))
    
    


def conv_with_autoencoding(features,targets):
    '''
    This implements autoencoding 
    '''
    features = features.copy()
    targets = targets.copy()
    
    cat_columns = ['AREA', 'PROBLEM', 'TOPIC']
    for colname in cat_columns:
        prefix = colname
        factor_features = pandas.get_dummies(features[colname])
        factor_features.columns = [prefix+'_'+x for x in factor_features.columns]
        features = pandas.concat([features.drop(colname,axis=1),factor_features],axis=1)
    
    print(features.columns)
    features = features.drop('STUDENT_ID',axis=1)
    # Attempt random fill values for preprocessing
    M = len(features.index)
    N = len(features.columns)
    ran = pandas.DataFrame(numpy.random.randn(M,N)*0.1, columns=features.columns, index=features.index)
    features.update(ran,overwrite=False)
    
    class_weights = {0: 1., 
                     1: 7.}
    
    
    input_shape = (features.shape[1],)
    batch_size = 32
    
    input_shape = Input(shape=(features.shape[1],))
    
    all_encoded = []
    
    encoded = Dense(128, activation='relu')(input_shape)
    all_encoded.append(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    all_encoded.append(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    all_encoded.append(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(features.shape[1], activation='sigmoid')(decoded)
    
    autoencoder = Model(input_shape, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(features, features,
                epochs=25,
                batch_size=32,
                shuffle=True,
                validation_data=(features, features))
    
    encoder = Model(input_shape,encoded)
    
    predicted_features = encoder.predict(features)
    print(predicted_features)
    
    
    new_features_shape = (predicted_features.shape[1],)
    
    model = Sequential()
    model.add(Dense(32,input_shape=new_features_shape))
    model.add(Dense(16))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    optimizer = keras.optimizers.Adadelta(lr=0.75, rho=0.95, epsilon=0.001, decay=1.)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    
    model.fit(predicted_features,targets,batch_size=32,epochs=50,class_weight=class_weights)
    score = model.evaluate(predicted_features, targets)
    predictions = model.predict(predicted_features)
    
    print(score)
    print(predictions)
    print(sklearn.metrics.f1_score(targets,pandas.Series(predictions.flatten()).map(int)))
    
    
    

def DKT_PredictionEncoder(refactored_sessions,max_seq_length):
    
    '''
    Used on the session by session data to create new features for each session
    in the other deep learning models via DKT
    
    Model adapted from base models of research papers provide fast prediction and training
    '''
    
    # The scores will be used as features in other models
    # no class weighting
    input_shape = (max_seq_length,)
    features,targets = refactored_sessions['features'],refactored_sessions['targets']
    
    model = Sequenital()
    model.add(Masking(mask_value=0.5,input_shape=input_shape))
    model.add(LSTM(32,return_sequences=True))
    #model.add(LSTM(16,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    
    optimizer = keras.optimizers.Adadelta(lr=0.75, rho=0.95, epsilon=0.001, decay=1.)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    
    model.fit(features,targets,batch_size=32,epochs=50)
    
    return model,model.predict(features)
    
    
    
    
def conv_model_with_DT_features(features,targets):
    '''
    # the basic premise of this model is to use stacking to build 
    # a convolutional deep learning model, motivated from Yang, H., & Cheung, L. P. (2018),
    where they used the same technique to measure student differences. In this case, 
    I use different tree models with different predictions to build quantile based features (tree splits essentialy)
    into a deep conv model.
    
    # TODO: do the same with cluster assignments
    
    '''
    
    ori_features,ori_targets = features.copy(),targets.copy()
    
    features = features.copy()
    targets = targets.copy()
    
    cat_columns = ['AREA', 'PROBLEM', 'TOPIC']
    for colname in cat_columns:
        prefix = colname
        factor_features = pandas.get_dummies(features[colname])
        factor_features.columns = [prefix+'_'+x for x in factor_features.columns]
        features = pandas.concat([features.drop(colname,axis=1),factor_features],axis=1)
    
    print(features.columns)
    features = features.drop('STUDENT_ID',axis=1)
    M = len(features.index)
    N = len(features.columns)
    # want disperion so as not to affect quantiles as much
    ran = pandas.DataFrame(numpy.random.randn(M,N)*2., columns=features.columns, index=features.index)
    features.update(ran,overwrite=False)
    
    class_weights = [
                    {0: 1. , 1: 1.},
                    {0: 1. , 1: 2.},
                    {0: 1. , 1: 5.},
                    {0: 1. , 1: 50.}
                    ]
    
    max_depths = [3,5,9]
    
    new_features = []
    for class_weight in class_weights:
        for max_depth in max_depths:
            print(max_depth,class_weight)
            model = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth,
                                                        class_weight=class_weight)
            model.fit(features,targets)
            predictions = model.predict(features)
            
            new_features.append(predictions)
            
    new_features = pandas.DataFrame(numpy.array(new_features).transpose(),index=ori_features.index)
    columns = ['DT_%s' % (str(i+1)) for i in range(len(new_features.columns))]
    new_features.columns = columns
    ori_features = ori_features.join(new_features)
    
    
    return basic_conv_model(ori_features,ori_targets)


# Dynamic Value Memory Network Code - Preprocess data then run MxNet model
# in_dir = '/Users/andrewgabriel/GaTech/EdTech/Project/DATA/SessionGrouped/'
def prepare_DKVMN_student(data_dict,student_id,problem_map):
    '''
    Use this function to make the data for running MxNet on a GPU for Dynamic Key Value
    Memory Networks as feature encoders. See https://github.com/jennyzhang0215/DKVMN
    '''
    keys = data_dict['ordered_keys']
    outlines = [] # list of length 3 lists to write to a file for MxNet
    
    def map_outcome(item):
        if item == 'INCORRECT':
            return 0
        elif item == 'CORRECT':
            return 1
        return None
    
    for key in keys:
        session_df  = data_dict['SessionData'][str(key)]
        session_df = pandas.DataFrame(session_df)
        problem_numbers = session_df['Problem Name'].map(lambda x: problem_map[x]).tolist()
        outcomes = session_df['Outcome'].map(map_outcome).tolist()
        outlines.append(str(len(problem_numbers)))
        outlines.append(','.join(map(str,problem_numbers)))
        outlines.append(','.join(map(str,outcomes)))
        
    return outlines
        
        
            
            
def prepare_DKVMN_data():
    in_dir  = '/Users/andrewgabriel/GaTech/EdTech/Project/DATA/SessionGrouped/'
    out_dir = '/Users/andrewgabriel/GaTech/EdTech/Project/DATA/MxNetData/'
    try:
        os.mkdir(out_dir)
    except:
        pass
    # get all the problems in the training set
    
    files = os.listdir(in_dir)
    all_problems = set()
    for filename in files:
        data = json.load(open(in_dir+filename))
        for key in data['SessionData']:
            df = pandas.DataFrame(data['SessionData'][key])
            problems = df['Problem Name'].value_counts().index.tolist()
            all_problems.update(problems)
            
    all_problems = sorted(list(all_problems))
    all_problems = {all_problems[i]:i+1 for i in range(len(all_problems))}
    all_lines = []
    for filename in files:
        data = json.load(open(in_dir+filename))
        student_id = filename.split('.')[0]
        outlines = prepare_DKVMN_student(data,student_id,all_problems)
        all_lines.extend(outlines)
        
    with open(out_dir+'mxnet_data.csv','w') as file_handle:
        file_handle.write('\n'.join(all_lines))
    
    # TODO: random injections for NAN values generated
        
        
        
    
# EdTechProjectMilestone2

Descriptions of Models:

basic_conv_model: This function makes a convolutional model that can be the head of any stacked version of the
research paper models or other models created.

conv_with_autoencoding: This function makes an autoencoder of the features as motivated by research papers in dimension 
reduction via deep learning. Essentially a more simple CONV model but more advanced feature ETL via deep learning

DKT_PredictionEncoder: This function makes a classical/basic Deep Knowledge Tracing Model that is going to be used in stacking ML models for prediction, as stated in numerous research papers on the topic. 

conv_model_with_DT_features: This is motivated from a paper where Decision Tree predictions were used in preprocessing data
for a DKT, but adjusted to work for a supervised learning task. Numerous Trees were trained 

prepare_DKVMN_student,prepare_DKVMN_data: This is used to work with an implemenation of Dynamic Key Value Memory Networks
for building features for supervised learning. That model is extremely advanced. The github code for it is found at https://github.com/jennyzhang0215/DKVMN


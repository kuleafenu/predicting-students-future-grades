import pandas as pd
import numpy as np
import pickle


class Predictor:
    """
    This is a helper function to help you perform custom prediction.
    argruments:
    model_path: The path to where the trained model is stored. make sure to give the exact directory to the file.
    data_path: The path to the preprocessed and saved comma separated file.
    scaler: This takes in the StandardScaler variable
    """
    def __init__(self,model_path,data_path,scaler):
        self.model_path = model_path
        self.data_path = data_path
        self.scaler = scaler
        
    def load_model(self):

        with open(self.model_path,'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model
        
    def get_prediction(self,G1,G2,number_of_absences,sample_index=7):
        """
        This function returns the prediction of a student grade based on a given arguments(student attributes)
        
        age: age of the student
        studytime: Student student study time
        number_of_absences: The number of times the student absented his/herself from school
        sample_index: index used to select data sample
        """

        
        model = self.load_model()

        data = pd.read_csv(self.data_path)    
        selected_sample = data.iloc[[sample_index]]
        
        selected_sample.loc[:,'G1'] = G1
        selected_sample.loc[:,'G2'] = G2
        selected_sample.loc[:,'absented_1'] = number_of_absences
        
        prediction = int(np.round(model.predict(self.scaler.transform(selected_sample.values))))   
        
        return prediction
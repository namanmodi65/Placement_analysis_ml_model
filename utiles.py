import joblib
import numpy as np

def preprocess(Age,Gender,Stream,Intership,CGPA,Backlog,Hostel):
    if(Hostel == 'Yes' or Hostel =='yes'):
        Hostel = 1
    else:
        Hostel = 0
    StreamNumber = 1
    if(Stream == 'Computer Science' or Stream == 'CS' ):
        StreamNumber=2
    if(Stream == 'Information Technology' or Stream == 'IT' ):
        StreamNumber=3
    if(Stream == 'Mechanical' ):
        StreamNumber=4
    if(Stream == 'Electrical' ):
        StreamNumber=5
    if(Stream == 'Civil' ):
        StreamNumber=6

    if(Gender == 'Male'):
        Gender = 1
    else:
        Gender = 0

    test_data = np.array([[int(Age),Gender,StreamNumber,int(Intership),int(CGPA),int(Backlog),Hostel]])
    trained_model = joblib.load('model_1.pkl')
    print(test_data)   
    prediction = trained_model.predict(test_data)
    print("Prediction :",prediction) 
    return prediction     
                    

from django.shortcuts import render

# Create your views here.
def about(request):
    return render(request,"about.html")
def register(request):
    if(request.method=="POST"):
        data=request.POST
        firstname=data.get("textfirstname")
        lastname=data.get("textlastname")
        if("buttonsubmit"in request.POST):
            result=firstname+" "+lastname+" Registration success"
            return render(request,"register.html",context={"result":result})
         
    return render(request,"register.html")
def index(request):
    return render(request,"index.html")

def about(request):
    return render(request, "about.html")



import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def heart(request):
    if request.method == "POST":
        data = request.POST
        Inputs = {
                'Age': int(data.get('textAge')),
                'Sex': int(data.get('textSex')),
                'Chestpain': float(data.get('textChestpain')),
                'Trestbps': float(data.get('textTrestbps')),
                'Cholestrol': float(data.get('textCholestrol')),
                'Fbs': int(data.get('textBloodsugar')),
                'Restecg': float(data.get('textRestecg')),
                'Thalach': float(data.get('textThalach')),
                'Exang': int(data.get('textExang')),
                'Oldpeak': float(data.get('textOldpeak')),
                'Slope': float(data.get('textSlope')),
                'Ca': float(data.get('textCa')),
                'Thalassemia': float(data.get('textThalassemia')),
            }
        if "buttonpredict" in request.POST:
            path = "C:\\Users\\rohan\\Desktop\\24_heartattackprediction\\heart .csv"
            dataset = pd.read_csv(path)
            inputs = dataset.drop('target', axis=1)
            output = dataset['target']

            # Train-test split
            x_train, x_test, y_train, y_test = train_test_split(inputs, output, train_size=0.8, random_state=42)

            # Scale data
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)

            # Train model
            model = SVC()
            model.fit(x_train, y_train)

            #inputs for prediction
            input_array = [
                Inputs['Age'], Inputs['Sex'], Inputs['Chestpain'], Inputs['Trestbps'],
                Inputs['Cholestrol'], Inputs['Fbs'], Inputs['Restecg'], Inputs['Thalach'],
                Inputs['Exang'], Inputs['Oldpeak'], Inputs['Slope'], Inputs['Ca'],
                Inputs['Thalassemia']
            ]
            input_array = [input_array]
            input_scaled = sc.transform(input_array)
            result = model.predict(input_scaled)
            return render(request, "heart.html", context={"result": "status:" + str(result[0])})

    return render(request, "heart.html")
  
def home(request):
    return render(request,"home.html")
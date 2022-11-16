#pip install Flask
from io import BytesIO
import numpy as  np
from flask import Flask, request, render_template,send_file
import pickle
import joblib
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import jinja2
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')


app =Flask(__name__) 
my_loader=jinja2.ChoiceLoader([app.jinja_loader,
                               jinja2.FileSystemLoader(r'C:\Users\Acer\Desktop\project construction\templates')])
app.jinja_loader=my_loader
model = pickle.load(open('model.pkl','rb'))
#model.predict(xtest)
a= joblib.load('Column1')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/vis',methods=['POST'])
def vis():
    return render_template('vis.html')

@app.route('/success',methods=['POST'])
def success():
    if request.method=='POST':
        f=request.files['file']
        read=pd.read_excel(f)
        data=pd.read_excel(f)
        LE=LabelEncoder()
        data['TYPE OF JOB']=LE.fit_transform(data['TYPE OF JOB'])
        data['BMI CAT']=LE.fit_transform(data['BMI CAT'])
        data['cat_age']=LE.fit_transform(data['cat_age'])
        data1=data.loc[:,['%HRR','ACC','BODY_TEMP','BMI CAT','cat_age','TYPE OF JOB','TOTAL_WORKING_Hr']]
        pred=model.predict(data1)
        read['Performance']=pred
        read.to_csv('result.csv',index=False) 
        return render_template('data.html',Z="LAST COLUMN SHOWING PERFORMANCE",Y=read.to_html())
@app.route('/visualization/')
def visualization():
    fig, axes = plt.subplots(2,2,figsize=(18,9)) 
    df=pd.read_csv('result.csv')
    perf=df['Performance']
    sns.countplot(perf)
    sns.countplot(df['cat_age'],ax=axes[0,0])
    axes[0,1].scatter(df['%HRR'],df['ACC'])
    sns.heatmap(df.corr(),cmap="Blues",ax=axes[1,0])
    canvas = FigureCanvasAgg(fig)
    img=BytesIO()
    fig.savefig(img)
    img.seek(0)
    return send_file(img,mimetype='image/png')
if __name__ == '__main__':
    app.run(host="127.0.0.9", port=8080, debug=True)
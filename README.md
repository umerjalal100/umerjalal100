import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import RepeatVector, LSTM
from tensorflow.keras.layers import Input,GRU, Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import streamlit as st
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
dff = pd.read_csv("My_Work .csv",encoding="ANSI")
dff=dff.drop(range(0,11))
dff.columns=dff.iloc[0]
dff=dff.iloc[1:]
dff.rename({"Code":"HS_06",'Exported value in 2003': '2003',
       'Exported value in 2004' :'2004', 'Exported value in 2005': '2005',
       'Exported value in 2006': '2006', 'Exported value in 2007': '2007',
       'Exported value in 2008': '2008', 'Exported value in 2009': '2009',
       'Exported value in 2010': '2010', 'Exported value in 2011': '2011',
       'Exported value in 2012': '2012', 'Exported value in 2013': '2013',
       'Exported value in 2014':'2014', 'Exported value in 2015':'2015',
       'Exported value in 2016':'2016', 'Exported value in 2017':'2017',
       'Exported value in 2018':'2018', 'Exported value in 2019':'2019',
       'Exported value in 2020':'2020', 'Exported value in 2021':'2021'}, axis = 1, inplace= True)
dff["HS_06"]=dff["HS_06"].str[1:]
dff["HS_02"]=dff["HS_06"].str[:-4]
dff=dff[['HS_02','HS_06','Product label', '2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021']]

df=dff.drop("2021", axis=1)
df=df.sort_values(by=["HS_02"], ascending=True).reset_index(drop =True, inplace= False).copy()
df=pd.concat([df.iloc[:,0:3],df.iloc[:,3:].astype(int)],axis=1)
Sec_72_96=df.iloc[3254:5103]
Sec_39_40=df.iloc[1605:1814]
Sec_69_70=df.iloc[3114:3213]
Engg=pd.concat([Sec_39_40,Sec_69_70,Sec_72_96]) # Sec = 39,40,69,70,72 to 96
Total_eng=Engg.sum()
Total_all=df.sum()

Iron_Steel=Engg[Engg.HS_02.str.contains('72')==True] 
exports=pd.read_csv("Exports.csv",encoding="ANSI")
exports.rename({"Code":"HS_06",'Exported value in 2003': '2003',
       'Exported value in 2004' :'2004', 'Exported value in 2005': '2005',
       'Exported value in 2006': '2006', 'Exported value in 2007': '2007',
       'Exported value in 2008': '2008', 'Exported value in 2009': '2009',
       'Exported value in 2010': '2010', 'Exported value in 2011': '2011',
       'Exported value in 2012': '2012', 'Exported value in 2013': '2013',
       'Exported value in 2014':'2014', 'Exported value in 2015':'2015',
       'Exported value in 2016':'2016', 'Exported value in 2017':'2017',
       'Exported value in 2018':'2018', 'Exported value in 2019':'2019',
       'Exported value in 2020':'2020', 'Exported value in 2021':'2021'}, axis = 1, inplace= True)
imports=pd.read_csv("Imports.csv",encoding="ANSI")
imports.rename({"Code":"HS_06",'Imported value in 2003': '2003',
       'Imported value in 2004' :'2004', 'Imported value in 2005': '2005',
       'Imported value in 2006': '2006', 'Imported value in 2007': '2007',
       'Imported value in 2008': '2008', 'Imported value in 2009': '2009',
       'Imported value in 2010': '2010', 'Imported value in 2011': '2011',
       'Imported value in 2012': '2012', 'Imported value in 2013': '2013',
       'Imported value in 2014':'2014', 'Imported value in 2015':'2015',
       'Imported value in 2016':'2016', 'Imported value in 2017':'2017',
       'Imported value in 2018':'2018', 'Imported value in 2019':'2019',
       'Imported value in 2020':'2020', 'Imported value in 2021':'2021'}, axis = 1, inplace= True)

Actual_year=list(imports.columns[3:11])
for j in range(8):
    get= Actual_year[j]
    get=str(get)

Mode = st.radio(
     "User Trend Analysis Options",
     ('Imports','Exports'))

Pred = st.sidebar.select_slider('Predicted Year Range: ', options=list(imports.iloc[1].index[11:]))

if Mode=="Imports":
    im=imports

    fig,ax=plt.subplots(figsize=(14,5))
    ax.plot(im.iloc[1]['2003':get].index, im.iloc[1]['2003':get], label = "Actual "+Mode)
    ax.scatter(im.iloc[1]['2003':get].index, im.iloc[1]['2003':get])
    ax.plot(im.iloc[1]['2011':Pred].index, im.iloc[1]['2011':Pred], linestyle = 'dashed', 
            label = "Predicted "+Mode)
    ax.scatter(im.iloc[1]['2011':Pred].index, im.iloc[1]['2011':Pred])
    ax.set_xlabel("Years")
    ax.set_ylabel('Engg. Goods '+Mode+" (US$:'000')")
    ax.tick_params(left=False,bottom=False)
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax.legend(loc="best")
    st.pyplot(fig)
    
    convert=im.iloc[1,1:]
    convert=convert.to_frame().reset_index().iloc[1:]
    convert.rename({"index":"Years", 1: Mode}, axis = 1, inplace= True)
    #st.dataframe(convert)
    st.sidebar.download_button('Download Complete Data',convert.to_csv(index=False),mime='text/csv')
    
elif Mode=='Exports':
    im=exports

    fig,ax=plt.subplots(figsize=(14,5)) 
    ax.plot(im.iloc[1]['2003':get].index, im.iloc[1]['2003':get], label = "Actual "+Mode)
    ax.scatter(im.iloc[1]['2003':get].index, im.iloc[1]['2003':get])
    ax.plot(im.iloc[1]['2011':Pred].index, im.iloc[1]['2011':Pred], linestyle = 'dashed',
            label = "Predicted "+Mode)
    ax.scatter(im.iloc[1]['2011':Pred].index, im.iloc[1]['2011':Pred])
    ax.set_xlabel("Years")
    ax.set_ylabel('Engg. Goods '+Mode+" (US$:'000')")
    ax.tick_params(left=False,bottom=False)
    for i in ax.spines:
        ax.spines[i].set_visible(False)
    ax.legend(loc="best")
    st.pyplot(fig)

    convert1=im.iloc[1,1:]
    convert1=convert1.to_frame().reset_index().iloc[1:]
    convert1.rename({"index":"Years", 1: Mode}, axis = 1, inplace= True)
    #st.dataframe(convert1)
    st.sidebar.download_button('Download Complete Data',convert1.to_csv(index=False),mime='text/csv') 
    


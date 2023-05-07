#!/usr/bin/env python
# coding: utf-8

# # Import required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
pd.pandas.set_option('display.max_columns',None)
# sns.set_style("whitegrid")


# # Load Dataset

# In[2]:


data=pd.read_csv(r"C:\Users\Dell\Desktop\data set\project\Property_Price_Train.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data1=pd.read_csv(r"C:\Users\Dell\Desktop\data set\project\Property_Price_Test.csv")


# In[7]:


data1


# In[8]:


data1.head()


# # Check information of Dataset

# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


# remember there way always be a chance pf data leakage so we need to split the data first and then apply feature engineering 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,data["Sale_Price"],test_size=0.1, random_state=0)


# In[12]:


x_train.shape,x_test.shape


# In[13]:


#let us capture all the nan values 
#first lets handle categorical features wich are missing 
feature_nan=[feature for feature in data.columns if data [feature].isnull().sum()>1 and data[feature].dtypes=="O"]
for feature in feature_nan:
    print("{}:{}% missing values".format(feature ,np.round(data[feature].isnull().mean(),4)))


# In[14]:


#replace missing value with a new lable 
def replace_cat_feature(data,feature_nan):
    data=data.copy()
    data[feature_nan]=data[feature_nan].fillna("missing")
    return data
data=replace_cat_feature(data,feature_nan)
data[feature_nan].isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# # Check if missing value is present or not for train

# In[15]:


#let us capture all the nan values
#first lets handle categorical fitcher which are missing 
#features_nan=[feature for featurein data.columns if data[feature].isnull().sum()>1 and data.]


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


#drop_col=["Id","Miscellaneous_Feature","Fence_Quality","Pool_Quality","Lane_Type"]# value have a 50% missing value


# In[17]:


#data.drop(drop_col,axis=1,inplace=True)


# In[18]:


data.Lot_Extent.value_counts()


# In[19]:


data.Lot_Extent=data.Lot_Extent.fillna(46.0 )


# In[20]:


data.Brick_Veneer_Type.value_counts()


# In[21]:


data.Brick_Veneer_Type=data.Brick_Veneer_Type.fillna("BrkCmn")


# In[22]:


data.Brick_Veneer_Area.value_counts()


# In[23]:


data.Brick_Veneer_Area=data.Brick_Veneer_Area.fillna(119)


# In[24]:


data.Basement_Height.value_counts()


# In[25]:


data.Basement_Height=data.Basement_Height.fillna("Fa")


# In[26]:


data.Basement_Condition.value_counts()


# In[27]:


data.Basement_Condition=data.Basement_Condition.fillna("Po")


# In[28]:


data.Exposure_Level.value_counts()


# In[29]:


data.Exposure_Level=data.Exposure_Level.fillna("Mn")


# In[30]:


data.BsmtFinType1.value_counts()


# In[31]:


data.BsmtFinType1=data.BsmtFinType1.fillna("LwQ")


# In[32]:


data.BsmtFinType2.value_counts()


# In[33]:


data.BsmtFinType2=data.BsmtFinType2.fillna("GLQ")


# In[34]:


data.Electrical_System.value_counts()


# In[35]:


data.Electrical_System=data.Electrical_System.fillna("Mix")


# In[36]:


data.Garage.value_counts()


# In[37]:


data.Garage=data.Garage.fillna("2Types")


# In[38]:


data.Garage_Built_Year.value_counts()


# In[39]:


data.Garage_Built_Year=data.Garage_Built_Year.fillna(1933)


# In[40]:


data.Garage_Finish_Year.value_counts()


# In[41]:


data.Garage_Finish_Year=data.Garage_Finish_Year.fillna("Fin")


# In[42]:


data.Garage_Quality.value_counts()


# In[43]:


data.Garage_Quality=data.Garage_Quality.fillna("Po")


# In[44]:


data.Garage_Condition.value_counts()


# In[45]:


data.Garage_Condition=data.Garage_Condition.fillna("Ex")


# In[46]:


data.Month_Sold.value_counts()


# In[47]:


data.Sale_Price.value_counts()


# # Check Datatypes

# In[48]:


data.dtypes


# In[49]:


data.info()


# # Check information of test Dataset

# In[50]:


#let us capture all the nan values 
#first lets handle categorical features wich are missing 
feature_nan=[feature for feature in data1.columns if data1[feature].isnull().sum()>1 and data1[feature].dtypes=="O"]
for feature in feature_nan:
    print("{}:{}% missing values".format(feature ,np.round(data1[feature].isnull().mean(),4)))


# In[51]:


#replace missing value with a new lable 
def replace_cat_feature(data1,feature_nan):
    data=data1.copy()
    data[feature_nan]=data[feature_nan].fillna("missing")
    return data
data1=replace_cat_feature(data1,feature_nan)
data1[feature_nan].isnull().sum()


# In[ ]:





# In[ ]:





# In[52]:


#data1.info()


# In[53]:


#drop_col=["Id","Miscellaneous_Feature","Fence_Quality","Pool_Quality","Fireplace_Quality","Lane_Type","Utility_Type"]# value have a 50% missing value


# In[54]:


#data1.drop(drop_col,axis=1,inplace=True)


# In[55]:


#data1.Sale_Condition.value_counts()


# # Check if missing value is present or not for test¶

# In[56]:


data1.Zoning_Class.value_counts()


# In[57]:


data1.Zoning_Class=data.Zoning_Class.fillna("RHD")


# In[58]:


data1.Lot_Extent.value_counts()


# In[59]:


data1.Lot_Extent=data.Lot_Extent.fillna(140)


# In[60]:


data1.Exterior1st.value_counts()


# In[61]:


data1.Exterior1st=data.Exterior1st.fillna("CB")


# In[62]:


data1.Brick_Veneer_Type.value_counts()


# In[63]:


data1.Brick_Veneer_Type=data.Brick_Veneer_Type.fillna("BrkCmn")


# In[64]:


data1.Basement_Height.value_counts()


# In[65]:


data1.Basement_Height=data.Basement_Height.fillna("Fa")


# In[66]:


data1.Basement_Condition.value_counts()


# In[67]:


data1.Basement_Condition=data.Basement_Condition.fillna("Po")


# In[68]:


data1.BsmtFinType1.value_counts()


# In[69]:


data1.BsmtFinType1=data.BsmtFinType1.fillna("LwQ")


# In[70]:


data1.BsmtFinType2.value_counts()


# In[71]:


data1.BsmtFinType2=data.BsmtFinType2.fillna("GLQ")


# In[72]:


data1.BsmtUnfSF.value_counts()


# In[73]:


data1.BsmtUnfSF=data.BsmtUnfSF.fillna(1503)


# In[74]:


data1.Underground_Full_Bathroom.value_counts()


# In[75]:


data1.Underground_Full_Bathroom=data.Underground_Full_Bathroom.fillna(3)


# In[76]:


data1.Kitchen_Quality.value_counts()


# In[77]:


data1.Kitchen_Quality=data.Kitchen_Quality.fillna("Fa")


# In[78]:


data1.Functional_Rate.value_counts()


# In[79]:


data1.Functional_Rate=data.Functional_Rate.fillna("MS")


# In[80]:


data1.Garage.value_counts()


# In[81]:


data1.Garage=data.Garage.fillna("CarPort")


# In[82]:


data1.Garage_Finish_Year.value_counts()


# In[83]:


data1.Garage_Finish_Year=data.Garage_Finish_Year.fillna("Fin")


# In[84]:


data1.Garage_Area.value_counts()


# In[85]:


data1.Garage_Area=data.Garage_Area.fillna(682)


# In[86]:


data1.Garage_Condition.value_counts()


# In[87]:


data1.Garage_Condition=data.Garage_Condition.fillna("Ex")


# In[88]:


data1.Exterior2nd.value_counts()


# In[89]:


data1.Exterior2nd=data.Exterior2nd.fillna("Stone")


# In[90]:


data1.Brick_Veneer_Area.value_counts()


# In[91]:


data1.Brick_Veneer_Area=data.Brick_Veneer_Area.fillna(382)


# In[92]:


data1.Exposure_Level.value_counts()


# In[93]:


data1.Exposure_Level=data.Exposure_Level.fillna("Mn")


# In[94]:


data1.BsmtFinSF1.value_counts()


# In[95]:


data1.BsmtFinSF1=data.BsmtFinSF1.fillna(337)


# In[96]:


data1.BsmtFinSF2.value_counts()


# In[97]:


data1.BsmtFinSF2=data.BsmtFinSF2.fillna(344)


# In[98]:


data1.Total_Basement_Area.value_counts()


# In[99]:


data1.Total_Basement_Area=data.Total_Basement_Area.fillna(996)


# In[100]:


data1.Underground_Half_Bathroom.value_counts()


# In[101]:


data1.Underground_Half_Bathroom=data.Underground_Half_Bathroom.fillna(2)


# In[102]:


data1.Garage_Built_Year.value_counts()


# In[103]:


data1.Garage_Built_Year=data.Garage_Built_Year.fillna(1919)


# In[104]:


data1.Garage_Size.value_counts()


# In[105]:


data1.Garage_Size=data.Garage_Size.fillna(5)


# In[106]:


data1.Garage_Quality.value_counts()


# In[107]:


data1.Garage_Quality=data.Garage_Quality.fillna("Po")


# In[108]:


data1.Sale_Type.value_counts()


# In[109]:


data1.Sale_Type=data.Sale_Type.fillna("ConLw")


# In[110]:


categorical_features=[feature for feature in data.columns if data[feature].dtype=='0']
categorical_features


# # Dtype Conversion

# In[111]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[112]:


data.Road_Type = le.fit_transform(data.Road_Type)


# In[113]:


data.Land_Outline = le.fit_transform(data.Land_Outline)


# In[114]:


data.Property_Slope = le.fit_transform(data.Property_Slope)


# In[115]:


data.Condition1 = le.fit_transform(data.Condition1)


# In[116]:


data.House_Type = le.fit_transform(data.House_Type )


# In[117]:


data.Roof_Design = le.fit_transform(data.Roof_Design)


# In[118]:


data.Exterior1st = le.fit_transform(data.Exterior1st)


# In[119]:


data.Brick_Veneer_Type =le.fit_transform(data.Brick_Veneer_Type )


# In[120]:


data.Exterior_Material = le.fit_transform(data.Exterior_Material)


# In[121]:


data.BsmtFinType1 = le.fit_transform(data.BsmtFinType1)


# In[122]:


data.Heating_Type = le.fit_transform(data.Heating_Type)


# In[123]:


data.Air_Conditioning = le.fit_transform(data.Air_Conditioning)


# In[124]:


data.Kitchen_Quality = le.fit_transform(data.Kitchen_Quality)


# In[125]:


data.Functional_Rate = le.fit_transform(data.Functional_Rate)


# In[126]:


data.Garage = le.fit_transform(data.Garage)


# In[127]:


data.Garage_Finish_Year = le.fit_transform(data.Garage_Finish_Year)


# In[128]:


data.Garage_Condition = le.fit_transform(data.Garage_Condition)


# In[129]:


data.Sale_Condition =le.fit_transform(data.Sale_Condition)


# In[130]:


data.Foundation_Type =le.fit_transform(data.Foundation_Type)


# In[131]:


data.Basement_Height =le.fit_transform(data.Basement_Height)


# In[132]:


data.Basement_Condition =le.fit_transform(data.Basement_Condition)


# In[133]:


data.BsmtFinType2 =le.fit_transform(data.BsmtFinType2)


# In[134]:


data.Zoning_Class =le.fit_transform(data.Zoning_Class)


# In[135]:


data.Property_Shape =le.fit_transform(data.Property_Shape)


# In[136]:


data.Utility_Type =le.fit_transform(data.Utility_Type)


# In[137]:


data.Lot_Configuration =le.fit_transform(data.Lot_Configuration)


# In[138]:


data.Neighborhood =le.fit_transform(data.Neighborhood)


# In[139]:


data.House_Design =le.fit_transform(data.House_Design)


# In[140]:


data.Roof_Quality =le.fit_transform(data.Roof_Quality)


# In[141]:


data.Exterior2nd =le.fit_transform(data.Exterior2nd)


# In[142]:


data.Exposure_Level =le.fit_transform(data.Exposure_Level)


# In[143]:


data.Heating_Quality =le.fit_transform(data.Heating_Quality)


# In[144]:


data.Electrical_System =le.fit_transform(data.Electrical_System)


# In[145]:


data.Fireplace_Quality =le.fit_transform(data.Fireplace_Quality)


# In[146]:


data.Pavedd_Drive =le.fit_transform(data.Pavedd_Drive)


# In[147]:


data.Sale_Type =le.fit_transform(data.Sale_Type)


# In[148]:


data.Condition2 =le.fit_transform(data.Condition2)


# In[149]:


data.Exterior_Condition =le.fit_transform(data.Exterior_Condition)


# In[150]:


data.Garage_Quality =le.fit_transform(data.Garage_Quality)


# In[151]:


data.Lane_Type =le.fit_transform(data.Lane_Type)
data.Pool_Quality =le.fit_transform(data.Pool_Quality)
data.Fence_Quality =le.fit_transform(data.Fence_Quality)
data.Miscellaneous_Feature =le.fit_transform(data.Miscellaneous_Feature)


# In[152]:


data.info()


# # Dtype Conversion for test

# In[153]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1.Road_Type = le.fit_transform(data.Road_Type)
data1.Land_Outline = le.fit_transform(data.Land_Outline)
data1.Property_Slope = le.fit_transform(data.Property_Slope)
data1.Condition1 = le.fit_transform(data.Condition1)
data1.House_Type = le.fit_transform(data.House_Type )
data1.Roof_Design = le.fit_transform(data.Roof_Design)
data1.Exterior1st = le.fit_transform(data.Exterior1st)
data1.Brick_Veneer_Type =le.fit_transform(data.Brick_Veneer_Type )
data1.Exterior_Material = le.fit_transform(data.Exterior_Material)
data1.BsmtFinType1 = le.fit_transform(data.BsmtFinType1)
data1.Heating_Type = le.fit_transform(data.Heating_Type)
data1.Air_Conditioning = le.fit_transform(data.Air_Conditioning)
data1.Kitchen_Quality = le.fit_transform(data.Kitchen_Quality)
data1.Functional_Rate = le.fit_transform(data.Functional_Rate)
data1.Garage = le.fit_transform(data.Garage)
data1.Garage_Finish_Year = le.fit_transform(data.Garage_Finish_Year)
data1.Garage_Condition = le.fit_transform(data.Garage_Condition)
data1.Sale_Condition =le.fit_transform(data.Sale_Condition)
data1.Foundation_Type =le.fit_transform(data.Foundation_Type)
data1.Basement_Height =le.fit_transform(data.Basement_Height)
data1.Basement_Condition =le.fit_transform(data.Basement_Condition)
data1.BsmtFinType2 =le.fit_transform(data.BsmtFinType2)
data1.Zoning_Class =le.fit_transform(data.Zoning_Class)
data1.Property_Shape =le.fit_transform(data.Property_Shape)
data1.Utility_Type =le.fit_transform(data.Utility_Type)
data1.Lot_Configuration =le.fit_transform(data.Lot_Configuration)
data1.Neighborhood =le.fit_transform(data.Neighborhood)
data1.House_Design =le.fit_transform(data.House_Design)
data1.Roof_Quality =le.fit_transform(data.Roof_Quality)
data1.Exterior2nd =le.fit_transform(data.Exterior2nd)
data1.Exposure_Level =le.fit_transform(data.Exposure_Level)
data1.Heating_Quality =le.fit_transform(data.Heating_Quality)
data1.Electrical_System =le.fit_transform(data.Electrical_System)
data1.Fireplace_Quality =le.fit_transform(data.Fireplace_Quality)
data1.Pavedd_Drive =le.fit_transform(data.Pavedd_Drive)
data1.Condition2 =le.fit_transform(data.Condition2)
data1.Exterior_Condition =le.fit_transform(data.Exterior_Condition)
data1.Garage_Quality =le.fit_transform(data.Garage_Quality)
data1.Garage_Quality =le.fit_transform(data.Garage_Quality)
data1.Sale_Type =le.fit_transform(data.Sale_Type)
data1.Lane_Type =le.fit_transform(data.Lane_Type)
data1.Pool_Quality =le.fit_transform(data.Pool_Quality)
data1.Fence_Quality =le.fit_transform(data.Fence_Quality)
data1.Miscellaneous_Feature =le.fit_transform(data.Miscellaneous_Feature)


# In[154]:


data1.info()


# In[ ]:





# In[155]:


plt.figure(figsize=(10,5))
sns.heatmap(data.isnull())


# In[156]:


plt.figure(figsize=(10,5))
sns.heatmap(data1.isnull())


# In[157]:


data_corr=data.corr()
data_corr


# In[158]:


plt.figure(figsize=(30,15))
Heatmap=sns.heatmap(data_corr,linewidth=1,annot=True,cmap=plt.cm.Blues)
plt.title("Heatmap for correlation")
plt.show()


# In[159]:


y_train=data[['Sale_Price']]


# In[160]:


X_train=data.drop(['Sale_Price'],axis=1)


# In[161]:


from sklearn.linear_model import Lasso
feature_sel_model=SelectFromModel(Lasso(alpha=0.005,random_state=0))


# In[162]:


feature_sel_model.fit(X_train,y_train)


# In[163]:


feature_sel_model.get_support()


# In[ ]:





# # Handling Rare Categorical feature

# we will remove categorical variable that are present less then 1% of observation

# In[532]:


categorical_features=[feature for feature in data.columns if data[feature].dtype=='O']
categorical_features


# In[165]:


#sns.countplot("Sale_Price",data=data)


# # Check model before EDA

# In[166]:


data.Sale_Price.value_counts()


# In[511]:


x=data.iloc[:,2:]
y=data.iloc[:,-1]


# In[512]:


x.shape,y.shape


# In[513]:


x.head()


# In[514]:


y.head()


# # Split Train and Test Data

# In[515]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=101)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # 1. Linear Regression

# In[516]:


from sklearn import linear_model
lin=linear_model.LinearRegression()


# In[517]:


lin.fit(x_train,y_train)


# In[518]:


y_train.head(2)


# In[519]:


lin_p=lin.predict(x_test)
lin_p


# In[520]:


lin.coef_


# In[521]:


lin.intercept_  


# In[522]:


R2=lin.score(x_train,y_train)
R2


# In[523]:


Adj_R2=1-(((1-R2)*(1167-1))/(1167-80-1))
Adj_R2


# In[524]:


from sklearn import metrics


# In[525]:


mse=metrics.mean_squared_error(y_test,lin_p)
mse


# In[526]:


rmse=pow(mse,0.5)
rmse


# In[527]:


df1=pd.DataFrame({"Actual":y_test,"Predicted":lin_p})
df1


# In[528]:


sns.lmplot(x="Actual",y="Predicted",data=df1,fit_reg=False)
d_line=np.arange(df1.min().min(),df1.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # Lasso(L1)

# In[ ]:





# In[529]:


from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(x_train,y_train)


# In[530]:


l_p=lasso.predict(x_test)
l_p


# In[531]:


lasso.coef_    #feature selection 


# In[188]:


l_R2=lasso.score(x_train,y_train)
l_R2


# In[189]:


l_Adj_R2=1-(((1-l_R2)*(1167-1))/(1167-80-1))
l_Adj_R2


# In[190]:


l_mse=metrics.mean_squared_error(y_test,l_p)
l_mse


# In[191]:


l_df=pd.DataFrame({"Importances":list(lasso.coef_),"column":list(x_test)})
l_df


# In[192]:


n_df1=pd.DataFrame({"Actual_n":y_test,"Predicted_n":l_p})
n_df1


# In[193]:


sns.lmplot(x="Actual_n",y="Predicted_n",data=n_df1,fit_reg=False)
d_line=np.arange(n_df1.min().min(),n_df1.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # 3. Random Forest Regressor

# In[194]:


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()


# In[195]:


rfr.fit(x_train,y_train)


# In[196]:


rfr_p=rfr.predict(x_test)


# In[197]:


rf_rsq=rfr.score(x_train,y_train)
rf_rsq


# In[198]:


rf_adr=1-(((1-rf_rsq)*(1167-1))/(1167-75-1))
rf_adr


# In[199]:


rfr_MSE=metrics.mean_squared_error(y_test,rfr_p)
rfr_MSE


# In[200]:


rf_df=pd.DataFrame({"Actual":y_test,"Predicted":rfr_p})
rf_df


# In[201]:


sns.lmplot(x="Actual",y="Predicted",data=rf_df,fit_reg=False)
d_line=np.arange(rf_df.min().min(),rf_df.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # Boosting

# # 1.XG-Boost Regressor

# In[202]:


from xgboost import XGBRFRegressor
xgb=XGBRFRegressor()


# In[203]:


xgb.fit(x_train,y_train)


# In[204]:


xg_p=xgb.predict(x_test)


# In[205]:


xg_rsq=xgb.score(x_train,y_train)
xg_rsq


# In[206]:


xg_adr=1-(((1-xg_rsq)*(1167-1))/(1167-75-1))
xg_adr


# In[207]:


xg_mse=metrics.mean_squared_error(y_test,xg_p)
xg_mse


# In[208]:


xg_df=pd.DataFrame({"Actual":y_test,"Predicted":xg_p})
xg_df


# In[209]:


sns.lmplot(x="Actual",y="Predicted",data=xg_df,fit_reg=False)
d_line=np.arange(xg_df.min().min(),xg_df.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # 2 AdaBoostRegressor

# In[210]:


from sklearn.ensemble import AdaBoostRegressor
ad=AdaBoostRegressor()


# In[211]:


ad.fit(x_train,y_train)


# In[212]:


ad_p=ad.predict(x_test)


# In[213]:


ad_rsq=ad.score(x_train,y_train)
ad_rsq


# In[214]:


ad_adr=1-(((1-ad_rsq)*(1167-1))/(1167-75-1))
ad_adr


# In[215]:


ad_MSE=metrics.mean_squared_error(y_test,ad_p)
ad_MSE


# # Final result before EDA

# In[216]:


l1=["Linear Regression","Random Forest Regressor","XG-Boost Regressor","AdaBoostRegressor"]
l2=[R2,rf_rsq,xg_rsq,ad_rsq]
l3=[Adj_R2,rf_adr,xg_adr,ad_adr]
l4=[mse,rfr_MSE,xg_mse,ad_MSE]


# In[217]:


before_eda=pd.DataFrame({"Model_Name":l1,"R-square":l2,"Adj_R2":l3,"MSE":l4})
before_eda


# # EDA

# In[218]:


plt.figure(figsize=(20,10))
sns.boxplot(data=data)
plt.xticks(rotation = 70)
plt.show()


# In[219]:


data.hist(figsize=(25,25))
plt.show()


# # for building_class

# In[220]:


data.Building_Class.describe()


# In[221]:


sns.boxplot(y = data.Building_Class, data = data)


# In[222]:


plt.hist(data.Building_Class)


# In[223]:


sns.distplot(data.Building_Class)


# In[224]:


data.Building_Class.value_counts


# In[225]:


iqr=data.Building_Class.quantile(0.75)-data.Building_Class.quantile(0.25)
iqr


# In[226]:


lb=data["Building_Class"].mean()-2*data["Building_Class"].std()
ub=data["Building_Class"].mean()+2*data["Building_Class"].std()
print(lb,ub)


# In[227]:


data.loc[data["Building_Class"]>141.53061482833385,"Building_Class"]=141.53061482833385


# # for Zoning_Class

# In[228]:


#data.Zoning_Class.describe()


# In[229]:


#sns.boxplot(y = data.Zoning_Class, data = data)#catgorical data


# In[230]:


#plt.hist(data.Zoning_Class)


# In[231]:


#sns.distplot(data.Zoning_Class)


# In[232]:


#data.Zoning_Class.value_counts


# # for Lot_Extent

# In[233]:


sns.boxplot(data=data.Lot_Extent)
plt.show()


# In[234]:


plt.hist(data.Lot_Extent)


# In[235]:


sns.distplot(data.Lot_Extent)


# In[236]:


data.Lot_Extent.value_counts()


# In[237]:


iqr=data.Lot_Extent.quantile(0.75)-data.Lot_Extent.quantile(0.25)
iqr


# In[238]:


data["Lot_Extent"].describe()


# In[239]:


lb=data["Lot_Extent"].quantile(0.25)-iqr*3
ub=data["Lot_Extent"].quantile(0.75)+iqr*3
print(lb,ub)


# In[240]:


data.loc[data["Lot_Extent"]>178.0,"Lot_Extent"]=178.0


# # for Lot_Size

# In[241]:


sns.boxplot(y = data.Lot_Size, data = data)


# In[242]:


plt.hist(data.Lot_Size)


# In[243]:


sns.distplot(data.Lot_Size)


# In[244]:


data.Lot_Size.value_counts()


# In[245]:


iqr=data.Lot_Size.quantile(0.75)-data.Lot_Size.quantile(0.25)
iqr


# In[246]:


data["Lot_Size"].describe()


# In[247]:


lb=data["Lot_Size"].quantile(0.25)-iqr*3
ub=data["Lot_Size"].quantile(0.75)+iqr*3
print(lb,ub)


# In[248]:


data.loc[data["Lot_Size"]>23765.0,"Lot_Size"]=23765.0


# In[249]:


data.loc[data["Lot_Size"]<-4613.0,"Lot_Size"]=-4613.0


# ### For Road_Type

# In[250]:


#sns.boxplot(y = data.Road_Type, data = data)#categorical data


# In[251]:


#data.Road_Type.value_counts()


# ### Property_Shape

# In[252]:


#sns.boxplot(y = data.Property_Shape, data = data)#categrical data


# ### Land_Outline

# In[253]:


#sns.boxplot(y = data.Land_Outline, data = data)#categrical data


# In[254]:


#data.Land_Outline.value_counts()


# ### Utility_Type

# In[255]:


#sns.boxplot(y = data.Utility_Type, data = data)#categrical data


# In[256]:


#data.Utility_Type.value_counts()


# ### Lot_Configuration

# In[257]:


#sns.boxplot(y = data.Lot_Configuration, data = data)#categrical data


# In[258]:


#data.Lot_Configuration.value_counts()


# ### Property_Slope

# In[259]:


#sns.boxplot(y = data.Property_Slope, data = data)#categrical data


# In[260]:


#data.Property_Slope.value_counts()


# ### Neighborhood

# In[261]:


#sns.boxplot(y = data.Neighborhood, data = data)#no out layer


# ### Condition1

# In[262]:


#sns.boxplot(y = data.Condition1, data = data)#categrical data


# In[263]:


#data.Condition1.value_counts()


# ### Condition2 

# In[264]:


#sns.boxplot(y = data.Condition2, data = data)#categrical data


# In[265]:


#data.Condition2.value_counts()


# ### House_Type

# In[266]:


#sns.boxplot(y = data.House_Type, data = data)#categrical data


# In[267]:


#data.House_Type.value_counts()


# ### House_Design

# In[268]:


#sns.boxplot(y = data.House_Design, data = data)#categrical data


# ### Overall_Material

# In[269]:


#sns.boxplot(y = data.Overall_Material, data = data)#categrical data


# In[270]:


##plt.hist(data.Overall_Material)


# In[271]:


#sns.distplot(data.Overall_Material)


# In[272]:


#data.Overall_Material.value_counts()


# ### House_Condition

# In[273]:


#sns.boxplot(y = data.House_Condition, data = data)#categrical data


# ### Construction_Year

# In[274]:


#sns.boxplot(y = data.Construction_Year, data = data)#not doing out layer treatment in Construction_Year


# ### Remodel_Year

# In[275]:


#sns.boxplot(y = data.Remodel_Year, data = data)#not doing out layer treatment in Remodel_Year


# ### Roof_Design

# In[276]:


#sns.boxplot(y = data.Roof_Design, data = data)#categrical data


# In[277]:


#data.Roof_Design.value_counts()


# ### Roof_Quality

# In[278]:


#sns.boxplot(y = data.Roof_Quality, data = data)#categrical data


# In[279]:


#data.Roof_Quality.value_counts()


# ### Exterior1st

# In[280]:


#sns.boxplot(y = data.Exterior1st, data = data)#categrical data


# ### Exterior2nd

# In[281]:


#sns.boxplot(y = data.Exterior2nd, data = data)#categrical data


# ### Brick_Veneer_Type

# In[282]:


#sns.boxplot(y = data.Brick_Veneer_Type, data = data)#categrical data


# ### Brick_Veneer_Area

# In[283]:


#sns.boxplot(y = data.Brick_Veneer_Area, data = data)#categrical data


# ### Exterior_Material

# In[284]:


#sns.boxplot(y = data.Exterior_Material, data = data)#categrical data


# ### Exterior_Condition

# In[285]:


#sns.boxplot(y = data.Exterior_Condition, data = data)#categrical data


# In[286]:


#data.Exterior_Condition.value_counts()


# ### Foundation_Type

# In[287]:


#sns.boxplot(y = data.Foundation_Type, data = data)#categrical data


# ### Basement_Height

# In[288]:


#sns.boxplot(y = data.Basement_Height, data = data)#categrical data


# ### Basement_Condition

# In[289]:


#sns.boxplot(y = data.Basement_Condition, data = data)#categrical data


# In[290]:


#data.Basement_Condition.value_counts()


# ### Exposure_Level

# In[291]:


#sns.boxplot(y = data.Exposure_Level, data = data)#categrical data


# ### BsmtFinType1

# In[292]:


#sns.boxplot(y = data.BsmtFinType1, data = data)#no out layer


# ### BsmtFinType2

# In[293]:


#sns.boxplot(y = data.BsmtFinType2, data = data)#categrical data


# In[294]:


#data.BsmtFinType2.value_counts()


# ### BsmtFinSF1

# In[295]:


sns.boxplot(y = data.BsmtFinSF1, data = data)


# In[296]:


plt.hist(data.BsmtFinSF1)


# In[297]:


sns.distplot(data.BsmtFinSF1)


# In[298]:


data.BsmtFinSF1.value_counts()


# In[299]:


data["BsmtFinSF1"].describe()


# In[300]:


iqr=data.BsmtFinSF1.quantile(0.75)-data.BsmtFinSF1.quantile(0.25)
iqr


# In[301]:


lb=data["BsmtFinSF1"].quantile(0.25)-iqr*3
ub=data["BsmtFinSF1"].quantile(0.75)+iqr*3
print(lb,ub)


# In[302]:


data.loc[data["BsmtFinSF1"]>2848.0,"BsmtFinSF1"]=2848.0


# ### BsmtFinSF2

# In[303]:


#sns.boxplot(y = data.BsmtFinSF2, data = data)#categrical data


# In[304]:


#data["BsmtFinSF2"].describe()


# ### BsmtUnfSF

# In[305]:


sns.boxplot(data=data.BsmtUnfSF)
plt.show()


# In[306]:


plt.hist(data.BsmtUnfSF)


# In[307]:


sns.distplot(data.BsmtUnfSF)


# In[308]:


data.BsmtUnfSF.value_counts()


# In[309]:


iqr=data.BsmtUnfSF.quantile(0.75)-data.BsmtUnfSF.quantile(0.25)
iqr


# In[310]:


data["BsmtUnfSF"].describe()


# In[311]:


lb=data["BsmtUnfSF"].mean()-2*data["BsmtUnfSF"].std()
ub=data["BsmtUnfSF"].mean()+2*data["BsmtUnfSF"].std()
print(lb,ub)


# In[312]:


data.loc[data["BsmtUnfSF"]>1451.2840977690635,"BsmtUnfSF"]=1451.2840977690635


# ### Total_Basement_Area

# In[313]:


sns.boxplot(y = data.Total_Basement_Area, data = data)


# In[314]:


plt.hist(data.Total_Basement_Area)


# In[315]:


sns.distplot(data.Total_Basement_Area)


# In[316]:


data.Total_Basement_Area.value_counts()


# In[317]:


iqr=data.Total_Basement_Area.quantile(0.75)-data.Total_Basement_Area.quantile(0.25)
iqr


# In[318]:


data["Total_Basement_Area"].describe()


# In[319]:


lb=data["Total_Basement_Area"].quantile(0.25)-iqr*3
ub=data["Total_Basement_Area"].quantile(0.75)+iqr*3
print(lb,ub)


# In[320]:


data.loc[data["Total_Basement_Area"]>2807.5,"Total_Basement_Area"]=2807.5


# In[321]:


data.loc[data["Total_Basement_Area"]<-713.5,"Total_Basement_Area"]=-713.5


# ### Heating_Type

# In[322]:


#sns.boxplot(y = data.Heating_Type, data = data)#categrical data


# In[323]:


#data.Heating_Type.value_counts()


# ### Heating_Quality

# In[324]:


#sns.boxplot(y = data.Heating_Quality, data = data)#categrical data


# In[325]:


#data.Heating_Quality.value_counts()


# ### Air_Conditioning

# In[326]:


#sns.boxplot(y = data.Air_Conditioning, data = data)#categrical data


# In[327]:


#data.Air_Conditioning.value_counts()


# ### Electrical_System

# In[328]:


#sns.boxplot(y = data.Electrical_System, data = data)#categrical data


# In[329]:


#data.Electrical_System.value_counts()


# ### First_Floor_Area

# In[330]:


sns.boxplot(y = data.First_Floor_Area, data = data)


# In[331]:


plt.hist(data.First_Floor_Area)


# In[332]:


sns.distplot(data.First_Floor_Area)


# In[333]:


data.First_Floor_Area.value_counts()


# In[334]:


iqr=data.First_Floor_Area.quantile(0.75)-data.First_Floor_Area.quantile(0.25)
iqr


# In[335]:


data["First_Floor_Area"].describe()


# In[336]:


lb=data["First_Floor_Area"].quantile(0.25)-iqr*3
ub=data["First_Floor_Area"].quantile(0.75)+iqr*3
print(lb,ub)


# In[337]:


data.loc[data["First_Floor_Area"]>2920.0,"First_Floor_Area"]=2920.0


# ### Second_Floor_Area

# In[338]:


#sns.boxplot(y = data.Second_Floor_Area, data = data)#categrical data


# In[339]:


#data["Second_Floor_Area"].describe()


# In[340]:


#data.Second_Floor_Area.value_counts()


# ### LowQualFinSF

# In[341]:


#sns.boxplot(y = data.LowQualFinSF, data = data)#categrical data


# In[342]:


#data["LowQualFinSF"].describe()


# In[343]:


#data.LowQualFinSF.value_counts()


# ### Grade_Living_Area

# In[344]:


sns.boxplot(y = data.Grade_Living_Area, data = data)


# In[345]:


plt.hist(data.Grade_Living_Area)


# In[346]:


sns.distplot(data.Grade_Living_Area)


# In[347]:


data.Grade_Living_Area.value_counts()


# In[348]:


iqr=data.Grade_Living_Area.quantile(0.75)-data.Grade_Living_Area.quantile(0.25)
iqr


# In[349]:


data["Grade_Living_Area"].describe()


# In[350]:


lb=data["Grade_Living_Area"].quantile(0.25)-iqr*3
ub=data["Grade_Living_Area"].quantile(0.75)+iqr*3
print(lb,ub)


# In[351]:


data.loc[data["Grade_Living_Area"]>3723.0,"Grade_Living_Area"]=3723.0


# ### Underground_Full_Bathroom

# In[352]:


#sns.boxplot(y = data.Underground_Full_Bathroom, data = data)#categrical data


# ### Underground_Half_Bathroom

# In[353]:


#sns.boxplot(y = data.Underground_Half_Bathroom, data = data)#categrical data


# In[354]:


#data.Underground_Half_Bathroom.value_counts()


# ### Full_Bathroom_Above_Grade                                                   

# In[355]:


#sns.boxplot(y = data.Full_Bathroom_Above_Grade, data = data)#categrical data


# ### Half_Bathroom_Above_Grade

# In[356]:


#sns.boxplot(y = data.Half_Bathroom_Above_Grade, data = data)#categrical data


# ### Bedroom_Above_Grade

# In[357]:


#sns.boxplot(y = data.Bedroom_Above_Grade, data = data)#categrical data


# In[358]:


#data.Bedroom_Above_Grade.value_counts()


# ### Kitchen_Above_Grade

# In[359]:


#sns.boxplot(y = data.Kitchen_Above_Grade, data = data)#categrical data


# In[360]:


#data.Kitchen_Above_Grade.value_counts()


# ### Kitchen_Quality

# In[361]:


#sns.boxplot(y = data.Kitchen_Quality, data = data)#categrical data


# In[362]:


#data.Kitchen_Quality.value_counts()


# ### Rooms_Above_Grade

# In[363]:


#sns.boxplot(y = data.Rooms_Above_Grade, data = data)#categrical data


# In[364]:


#data.Rooms_Above_Grade.value_counts()


# ### Functional_Rate

# In[365]:


#sns.boxplot(y = data.Functional_Rate, data = data)#categrical data


# In[366]:


#data.Functional_Rate.value_counts()


# ### Fireplaces

# In[367]:


#sns.boxplot(y = data.Fireplaces, data = data)#categrical data


# In[368]:


#data.Fireplaces.value_counts()


# ### Fireplace_Quality 

# In[369]:


#sns.boxplot(y = data.Fireplace_Quality, data = data)#no out layer


# ### Garage

# In[370]:


#sns.boxplot(y = data.Garage, data = data)#categrical data


# In[371]:


#data.Garage.value_counts()


# ### Garage_Built_Year

# In[372]:


#sns.boxplot(y = data.Garage_Built_Year, data = data)#cno out laye


# #### Garage_Finish_Year

# In[373]:


#sns.boxplot(y = data.Garage_Finish_Year, data = data)#no out layers


# ### Garage_Size

# In[374]:


#sns.boxplot(y = data.Garage_Size, data = data)#categrical data


# In[375]:


#data.Garage_Size.value_counts()


# ### Garage_Area

# In[376]:


sns.boxplot(y = data.Garage_Area, data = data)


# In[377]:


data.Garage_Area.value_counts()


# In[378]:


plt.hist(data.Garage_Area)


# In[379]:


sns.distplot(data.Garage_Area)


# In[380]:


iqr=data.Garage_Area.quantile(0.75)-data.Garage_Area.quantile(0.25)
iqr


# In[381]:


data["Garage_Area"].describe()


# In[382]:


lb=data["Garage_Area"].mean()-2*data["Garage_Area"].std()
ub=data["Garage_Area"].mean()+2*data["Garage_Area"].std()
print(lb,ub)


# In[383]:


data.loc[data["Garage_Area"]>892.2723619563408,"Garage_Area"]=892.2723619563408


# In[384]:


data.loc[data["Garage_Area"]<49.59721226472243,"Garage_Area"]=49.59721226472243


# ### Garage_Quality

# In[385]:


#sns.boxplot(y = data.Garage_Quality, data = data)#categrical data


# In[386]:


#data.Garage_Quality.value_counts()


# ### Garage_Condition

# In[387]:


#sns.boxplot(y = data.Garage_Condition, data = data)#categrical data


# In[388]:


#data.Garage_Condition.value_counts()


# ### Pavedd_Drive

# In[389]:


#sns.boxplot(y = data.Pavedd_Drive, data = data)#categrical data


# In[390]:


#data.Pavedd_Drive.value_counts()


# ### W_Deck_Area

# In[391]:


sns.boxplot(y = data.W_Deck_Area, data = data)


# In[392]:


plt.hist(data.W_Deck_Area)


# In[393]:


sns.distplot(data.W_Deck_Area)


# In[394]:


data.W_Deck_Area.value_counts()


# In[395]:


iqr=data.W_Deck_Area.quantile(0.75)-data.W_Deck_Area.quantile(0.25)
iqr


# In[396]:


data["W_Deck_Area"].describe()


# In[397]:


lb=data["W_Deck_Area"].mean()-2*data["W_Deck_Area"].std()
ub=data["W_Deck_Area"].mean()+2*data["W_Deck_Area"].std()
print(lb,ub)


# In[398]:


data.loc[data["W_Deck_Area"]>342.6090307634815,"W_Deck_Area"]=342.6090307634815


# In[399]:


data.loc[data["W_Deck_Area"]<-156.5776962133031,"W_Deck_Area"]=-156.5776962133031


# ### Open_Lobby_Area

# In[400]:


sns.boxplot(y = data.Open_Lobby_Area, data = data)


# In[401]:


plt.hist(data.Open_Lobby_Area)


# In[402]:


sns.distplot(data.Open_Lobby_Area)


# In[403]:


data.Open_Lobby_Area.value_counts()


# In[404]:


iqr=data.Open_Lobby_Area.quantile(0.75)-data.Open_Lobby_Area.quantile(0.25)
iqr


# In[405]:


data["Open_Lobby_Area"].describe()


# In[406]:


lb=data["Open_Lobby_Area"].mean()-2*data["Open_Lobby_Area"].std()
ub=data["Open_Lobby_Area"].mean()+2*data["Open_Lobby_Area"].std()
print(lb,ub)


# In[407]:


data.loc[data["Open_Lobby_Area"]>182.7470738982994,"Open_Lobby_Area"]=182.7470738982994


# In[408]:


data.loc[data["Open_Lobby_Area"]<-87.12327075387584,"Open_Lobby_Area"]=-87.12327075387584


# ### Enclosed_Lobby_Area

# In[409]:


sns.boxplot(y = data.Enclosed_Lobby_Area, data = data)


# In[410]:


plt.hist(data.Enclosed_Lobby_Area)


# In[411]:


sns.distplot(data.Enclosed_Lobby_Area)


# In[412]:


data.Enclosed_Lobby_Area.value_counts()


# In[413]:


iqr=data.Enclosed_Lobby_Area.quantile(0.75)-data.Enclosed_Lobby_Area.quantile(0.25)
iqr


# In[414]:


data["Enclosed_Lobby_Area"].describe()


# In[415]:


lb=data["Enclosed_Lobby_Area"].mean()-2*data["Enclosed_Lobby_Area"].std()
ub=data["Enclosed_Lobby_Area"].mean()+2*data["Enclosed_Lobby_Area"].std()
print(lb,ub)


# In[416]:


data.loc[data["Enclosed_Lobby_Area"]>147.28724030164415,"Enclosed_Lobby_Area"]=147.28724030164415


# In[417]:


data.loc[data["Enclosed_Lobby_Area"]<-98.126786214594,"Enclosed_Lobby_Area"]=-98.126786214594


# ### Three_Season_Lobby_Area

# In[418]:


#sns.boxplot(y = data.Three_Season_Lobby_Area, data = data)#categrical data


# In[419]:


#data.Three_Season_Lobby_Area.value_counts()


# ### Screen_Lobby_Area

# In[420]:


#sns.boxplot(y = data.Screen_Lobby_Area, data = data)#categrical data


# In[421]:


#data["Screen_Lobby_Area"].describe()


# In[422]:


#data.Screen_Lobby_Area.value_counts()


# ### Pool_Area

# In[423]:


#sns.boxplot(y = data.Pool_Area, data = data)#categrical data


# In[424]:


#data.Pool_Area.value_counts()


# ### Miscellaneous_Value 

# In[425]:


#sns.boxplot(y = data.Miscellaneous_Value, data = data)#categrical data


# In[426]:


#data.Miscellaneous_Value.value_counts()


# ### Month_Sold

# In[427]:


#sns.boxplot(y = data.Month_Sold, data = data)#no out layer


# ### Year_Sold

# In[428]:


#sns.boxplot(y = data.Year_Sold, data = data)#no out layer


# ### Sale_Type

# In[429]:


#sns.boxplot(y = data.Sale_Type, data = data)#categrical data


# In[430]:


#data.Sale_Type.value_counts()


# ### Sale_Condition

# In[431]:


#sns.boxplot(y = data.Sale_Condition, data = data)#categrical data


# In[432]:


#data.Sale_Condition.value_counts()


# ### Sale_Price

# In[433]:


#sns.boxplot(y = data.Sale_Price, data = data)


# In[434]:


#plt.hist


# In[435]:


#plt.hist(data.Sale_Price)


# In[436]:


#sns.distplot(data.Sale_Price)


# In[437]:


#data.Sale_Price.value_counts()


# In[438]:


#iqr=data.Sale_Price.quantile(0.75)-data.Sale_Price.quantile(0.25)
#iqr


# In[439]:


#data["Sale_Price"].describe()


# In[440]:


#lb=data["Sale_Price"].mean()-2*data["Sale_Price"].std()
#ub=data["Sale_Price"].mean()+2*data["Sale_Price"].std()
#print(lb,ub)


# In[441]:


#data.loc[data["Sale_Price"]>339873.9394805562,"Sale_Price"]=339873.9394805562


# In[442]:


data.skew()


# # Data Transformation

# ### For Lot_Extent

# In[443]:


#Check zero values present or not
data[data['Lot_Extent']==0]


# In[444]:


sns.distplot(data.Lot_Extent)


# In[445]:


data.Lot_Extent.skew()


# In[446]:


data.Lot_Extent = np.log(data.Lot_Extent)
data.Lot_Extent.skew()


# In[447]:


sns.distplot(data.Lot_Extent)


# ### Data Visualization

# In[448]:


Sale_Price =data["Sale_Price"]
Sale_Price 


# In[449]:


df=data.drop(["Sale_Price"],axis=1)
df.head(2)


# In[450]:


s_x=df
s_y=Sale_Price


# In[451]:


s_x.shape


# ### Splitting train and test

# In[452]:


sx_train,sx_test,sy_train,sy_test=train_test_split(s_x,s_y,test_size=0.2,random_state=101)


# In[453]:


sx_train.shape,sx_test.shape,sy_train.shape,sy_test.shape


# In[454]:


sy_train.info()


# ### GradientBoostingRegressor

# In[455]:


from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor()
model.fit(sx_train,sy_train)


# In[456]:


lin_p1=model.predict(sx_test)
lin_p1


# In[457]:


s_R2=model.score(sx_train,sy_train)
s_R2


# In[458]:


s_adjR2=1-(((1-s_R2)*(1167-1))/(1167-75-1))
s_adjR2


# In[459]:


s_mse=metrics.mean_squared_error(sy_test,lin_p1)
s_mse


# In[460]:


s_rmse=pow(s_mse,0.5)
s_rmse


# # Linear Regression

# In[461]:


from sklearn import linear_model
lin=linear_model.LinearRegression()


# In[462]:


lin.fit(sx_train,sy_train)


# In[463]:


lin_p1=lin.predict(sx_test)
lin_p1


# In[464]:


s_R2=lin.score(sx_train,sy_train)
s_R2


# In[465]:


s_adjR2=1-(((1-s_R2)*(1167-1))/(1167-75-1))
s_adjR2


# In[466]:


s_mse=metrics.mean_squared_error(sy_test,lin_p1)
s_mse


# In[467]:


s_rmse=pow(s_mse,0.5)
s_rmse


# In[468]:


df2=pd.DataFrame({"Actual":sy_test,"Predicted":lin_p1})
df2


# In[469]:


sns.lmplot(x="Actual",y="Predicted",data=df2,fit_reg=False)
d_line=np.arange(df2.min().min(),df2.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # XG boost

# In[470]:


from xgboost import XGBRFRegressor
xg1=XGBRFRegressor()


# In[471]:


xg1.fit(sx_train,sy_train)


# In[472]:


xg1_p=xg1.predict(sx_test)


# In[473]:


R_xg1=xg1.score(sx_train,sy_train)
R_xg1


# In[474]:


Adj_xg1=1-(((1-R_xg1)*(1167-1))/(1167-75-1))
Adj_xg1


# In[475]:


xg1_mse=metrics.mean_squared_error(sy_test,xg1_p)
xg1_mse 


# In[476]:


xg1_df=pd.DataFrame({"Actual":sy_test,"Predicted":xg1_p})
xg1_df


# In[477]:


sns.lmplot(x="Actual",y="Predicted",data=xg1_df,fit_reg=False)
d_line=np.arange(xg1_df.min().min(),xg1_df.max().max())
plt.plot(d_line,d_line,color="yellow",linestyle="-")
plt.show()


# # Random Forest Regression

# In[478]:


from sklearn.ensemble import RandomForestRegressor
rfr1=RandomForestRegressor()


# In[479]:


rfr1.fit(sx_train,sy_train)


# In[480]:


rfr1_p=rfr1.predict(sx_test)


# In[481]:


R_rf1=rfr1.score(sx_train,sy_train)
R_rf1


# In[482]:


Adj_rf1=1-(((1-R_rf1)*(1167-1))/(1167-75-1))
Adj_rf1


# In[483]:


Rf_mse1=metrics.mean_squared_error(sy_test,rfr1_p)
Rf_mse1


# In[484]:


Rf_df=pd.DataFrame({"Actual":sy_test,"Predicted":rfr1_p})
Rf_df


# In[485]:


sns.lmplot(x="Actual",y="Predicted",data=Rf_df,fit_reg=False)
d_line=np.arange(Rf_df.min().min(),Rf_df.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# # After EDA Final result

# In[486]:


list5=["RandomForestRegressor","LinearRegression","XGBRFRegressor"]
list6=[R_rf1,s_R2,R_xg1]
list7=[Adj_rf1,s_adjR2,Adj_xg1]
list8=[Rf_mse1,s_mse,xg1_mse]


# In[487]:


after_eda=pd.DataFrame({"Model_Name":list5,"R-square":list6,"Adj_R2":list7,"MSE":list8})
after_eda


# In[488]:


before_eda


# # Conclusion

# In[489]:


sns.lmplot(x="Actual",y="Predicted",data=Rf_df,fit_reg=False)
d_line=np.arange(Rf_df.min().min(),Rf_df.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()
sns.lmplot(x="Actual",y="Predicted",data=rf_df,fit_reg=False)
d_line=np.arange(rf_df.min().min(),rf_df.max().max())
plt.plot(d_line,d_line,color="green",linestyle="-")
plt.show()


# In[ ]:





# In[ ]:





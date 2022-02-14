import numpy as n
from scipy.misc import derivative # to be able to calculate derivative directly
import pandas as p # to be able to import data set
import math as m
#Find YTM QUESTION1
# x0 is 0, 'small' is to set a number very close to zero, and maximum is how
#number of iteration.
#define a function to use newton method to find the value of x
def newton(fn, x0, small, m): 
    xn=x0
    for n in range (0, m):
        Dvalue = derivative(fn, xn, dx=1e-10)#derivative value
        if Dvalue != 0 and abs(fn(xn)) > small:
            xn = xn - fn(xn)/Dvalue 
        else:
            return xn

            

mydata = p.read_csv("/Users/tianqiyue/Desktop/apm466data.csv")#the bond prices that I collected
data = p.DataFrame(mydata)#create a dataframe
#calculating dirty price, number of day before last payment from jan 10 is the last column of my table.
#number of coupon payment is two, accrual period is 182.5, face value is 1000.
lista=[]
for i in range(0,10):# creating a list with all of the dirty prices of all bonds for ten days.
        for n in range(1,11):
            couponrate=float(data.iloc[i,11]) #coupon rate, column 11 is the coupon payment
            day=int(data.iloc[i,14]) #number of days since the last coupon payment to jan 10
            day1=day+n-1 # add the value to the number of days since the last coupon payment to jan 10 to match with other dates ie jan 11, jan 12
            price=float(data.iloc[i,n]) # clean prices
            dirtyprice=100/2*couponrate*day1/182.5+price #dirty prices using the formula of accrued interest plus the clean price
            lista=lista+[dirtyprice] #append the dirty price to the list
# cut the list into pieces
cut=[lista[i:i + 10] for i in range(0, len(lista), 10)]
#convert list a into dataframe
data_withdirtyprice=p.DataFrame([cut[0],cut[1],cut[2],cut[3],cut[4],cut[5],cut[6],cut[7],cut[8],cut[9]],columns=["1","2",'3','4','5','6','7','8','9','10'])
data_withdirtyprice['coupon']=data['Coupon %']#insert coupon rate column to the new dirty price table
data_withdirtyprice['number of coupon payment left']=data['Periods left']#insert number of payment left to the new table
#define function used to calculate yield to maturity 
def f(periodsleftd,couponrate,dirtypriced):
    def f2(x):
        return (couponrate*100/2*((1-(1+x)**-periodsleftd)/x))+((100/(1+x)**periodsleftd))-dirtypriced
    return f2
#generate function for all the prices
listb=[]
for a in range(0,10):
    for o in range(0,10):
        couponrate=float(data_withdirtyprice.iloc[a,10])
        periodsleftd=float(data_withdirtyprice.iloc[a,11])
        dirtypriced=float(data_withdirtyprice.iloc[a,o])
        listb=listb+[f(periodsleftd,couponrate,dirtypriced)]
ytm1=[]
#find the ytm for all the functions
for h in range(0,100):
    ytm=newton(listb[h], 0.00001,1e-10,1000)*2
    ytm1=ytm1+[ytm]
#convert the ytm list into dataframe
cutytm=[ytm1[i:i+10] for i in range(0,len(ytm1),10)]
ytmtable=p.DataFrame([cutytm[0],cutytm[1],cutytm[2],cutytm[3],cutytm[4],cutytm[5],cutytm[6],cutytm[7],cutytm[8],cutytm[9]],columns=["1","2",'3','4','5','6','7','8','9','10'])
#plot the data
import matplotlib.pyplot as plt
plt.subplot(1,3,1)
x=data['Times to maturity'].tolist()#x axis
plt.xlabel('Times to maturity')
plt.ylabel('YTM')
plt.plot(x,ytmtable['1'].tolist() ,label='Jan10')
plt.plot(x,ytmtable['2'].tolist() ,label='Jan11')
plt.plot(x,ytmtable['3'].tolist() ,label='Jan12')
plt.plot(x,ytmtable['4'].tolist() ,label='Jan13')
plt.plot(x,ytmtable['5'].tolist() ,label='Jan14')
plt.plot(x,ytmtable['6'].tolist() ,label='Jan17')
plt.plot(x,ytmtable['7'].tolist() ,label='Jan18')
plt.plot(x,ytmtable['8'].tolist() ,label='Jan19')
plt.plot(x,ytmtable['9'].tolist() ,label='Jan20')
plt.plot(x,ytmtable['10'].tolist() ,label='Jan21')
plt.title('YTM curve')
plt.legend()

#Annual Spot rate QUESTION 2
#data_withdirtyorice is the dirty price table
# find the first spot rate for 10 dates
one = data_withdirtyprice.iloc[0].tolist()# the dirty price of the bound with 1 payment left for 10 days Jan10 - Jan21
spotrate1=[]
# since the first bond mature in 3 month, the T would be 1/2
for i in range(0,10):
#one[10] is the coupon rate, and spot rate1 = (facevalue+coupon payment)/dirty price -1 
    sprate=-m.log(one[i]/((100*one[10])/2+100))/0.5
    spotrate1=spotrate1+[sprate]
# find the second spot rate for 10 dates
two= data_withdirtyprice.iloc[1].tolist() # the dirty prices of the bound with 2 payment left for 10 days
spotrate2=[]
for i in range(0,10):
#Two[10] is the coupon rate and the formula for second spot rate is the following equation
    sprate2=-m.log((two[i]-(100*two[10])/2*m.exp(spotrate1[i]*0.5))/((100*two[10])/2+100))
    spotrate2=spotrate2+[sprate2]
# find the third spot rate for 10 dates
three= data_withdirtyprice.iloc[2].tolist() # the dirty prices of the bound with 3 payment left for 10 days
spotrate3=[]
for i in range(0,10):
    sprate3=-m.log((three[i]-100*three[10]/2*m.exp(spotrate1[i]*0.5)-(100*three[10])/2*m.exp(spotrate2[i]))/((100*three[10])/2+100))/1.5
    spotrate3=spotrate3+[sprate3]   
# find the forth spot rate for 10 dates
forth=data_withdirtyprice.iloc[3].tolist() # the dirty prices of the bound with 4 payment left for 10 days
spotrate4=[]
for i in range(0,10):
    sprate4=-m.log((forth[i]-100*forth[10]/2*m.exp(spotrate1[i]*0.5)-(100*forth[10])/2*m.exp(spotrate2[i])-(100*three[10])/2*m.exp(spotrate3[i]*1.5))/((100*forth[10])/2+100))/2
    spotrate4=spotrate4+[sprate4]  
# find the fifth spot rate for 10 dates
fifth=data_withdirtyprice.iloc[4].tolist() # the dirty prices of the bound with 5 payments left for 10 days
spotrate5=[]
for i in range(0,10):
    coupon5=fifth[10]*100/2
    sprate5=-m.log((fifth[i]-coupon5*m.exp(spotrate1[i]*0.5)-coupon5*m.exp(spotrate2[i])-coupon5*m.exp(spotrate3[i]*1.5)-coupon5*m.exp(spotrate4[i]*2))/((coupon5+100)))/2.5
    spotrate5=spotrate5+[sprate5]
# find the sixth spot rate for 10 dates
sixth=data_withdirtyprice.iloc[5].tolist() # the dirty prices of the bound with 6 payments left for 10 days
spotrate6=[]
for i in range(0,10):
    coupon6=sixth[10]*100/2
    sprate6=-m.log((sixth[i]-coupon6*m.exp(spotrate1[i]*0.5)-coupon6*m.exp(spotrate2[i])-coupon6*m.exp(spotrate3[i]*1.5)-coupon6*m.exp(spotrate4[i]*2)-coupon6*m.exp(spotrate5[i]*2.5))/((coupon6+100)))/3
    spotrate6=spotrate6+[sprate6] 
# find the seventh spot rate for 10 dates
seventh=data_withdirtyprice.iloc[6].tolist() # the dirty prices of the bound with 7 payments left for 10 days
spotrate7=[]
for i in range(0,10):
    coupon7=seventh[10]*100/2
    sprate7=-m.log((seventh[i]-coupon7*m.exp(spotrate1[i]*0.5)-coupon7*m.exp(spotrate2[i])-coupon7*m.exp(spotrate3[i]*1.5)-coupon7*m.exp(spotrate4[i]*2)-coupon7*m.exp(spotrate5[i]*2.5)-coupon7*m.exp(spotrate6[i]*3))/((coupon7+100)))/3.5
    spotrate7=spotrate7+[sprate7] 
# find the eighth spot rate for 10 dates
eighth=data_withdirtyprice.iloc[7].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate8=[]
for i in range(0,10):
    coupon8=eighth[10]*100/2
    sprate8=-m.log((eighth[i]-coupon8*m.exp(spotrate1[i]*0.5)-coupon8*m.exp(spotrate2[i])-coupon8*m.exp(spotrate3[i]*1.5)-coupon8*m.exp(spotrate4[i]*2)-coupon8*m.exp(spotrate5[i]*2.5)-coupon8*m.exp(spotrate6[i]*3)-coupon8*m.exp(spotrate7[i]*3.5))/((coupon8+100)))/4
    spotrate8=spotrate8+[sprate8] 
# find the ninth spot rate for 10 dates
ninth=data_withdirtyprice.iloc[8].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate9=[]
for i in range(0,10):
    coupon9=ninth[10]*100/2
    sprate9=-m.log((ninth[i]-coupon9*m.exp(spotrate1[i]*0.5)-coupon9*m.exp(spotrate2[i])-coupon9*m.exp(spotrate3[i]*1.5)-coupon9*m.exp(spotrate4[i]*2)-coupon9*m.exp(spotrate5[i]*2.5)-coupon9*m.exp(spotrate6[i]*3)-coupon9*m.exp(spotrate7[i]*3.5)-coupon9*m.exp(spotrate8[i]*4))/((coupon9+100)))/4.5
    spotrate9=spotrate9+[sprate9] 
# find the ninth spot rate for 10 dates
tenth=data_withdirtyprice.iloc[9].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate10=[]
for i in range(0,10):
    coupon10=tenth[10]*100/2
    sprate10=-m.log((tenth[i]-coupon10*m.exp(spotrate1[i]*0.5)-coupon10*m.exp(spotrate2[i])-coupon10*m.exp(spotrate3[i]*1.5)-coupon10*m.exp(spotrate4[i]*2)-coupon10*m.exp(spotrate5[i]*2.5)-coupon10*m.exp(spotrate6[i]*3)-coupon10*m.exp(spotrate7[i]*3.5)-coupon10*m.exp(spotrate8[i]*4)-coupon10*m.exp(spotrate9[i]*4.5))/((coupon10+100)))/5
    spotrate10=spotrate10+[sprate10]
#create a dataframe for spotrate
spottable=p.DataFrame([spotrate1,spotrate2,spotrate3,spotrate4,spotrate5,spotrate6,spotrate7,spotrate8,spotrate9,spotrate10],columns=["1","2",'3','4','5','6','7','8','9','10'])
plt.subplot(1,3,2)
x=['1','2','3','4','5','6','7','8','9','10']#x axis periods left to maturity
plt.xlabel('Periods (half year) to maturity', fontsize=10)
plt.ylabel('spot rate')
plt.plot(x,spottable['1'] ,label='Jan10')
plt.plot(x,spottable['2'] ,label='Jan11')
plt.plot(x,spottable['3'] ,label='Jan12')
plt.plot(x,spottable['4'],label='Jan13')
plt.plot(x,spottable['5'] ,label='Jan14')
plt.plot(x,spottable['6'],label='Jan17')
plt.plot(x,spottable['7'],label='Jan18')
plt.plot(x,spottable['8'] ,label='Jan19')
plt.plot(x,spottable['9'],label='Jan20')
plt.plot(x,spottable['10'] ,label='Jan21')
plt.title('Spot curve')
plt.legend()
plt.show()
#Qustion3 Foward rate
#spot table is the spotrate table derived from question 2
#calculating 1yr 1yr rates
oneyrspot=spottable.iloc[1].tolist()
twoyrspot=spottable.iloc[3].tolist()
forward1=[]
for i in range (0,10):
    forward1_1=twoyrspot[i]*2-oneyrspot[i]
    forward1=forward1+[forward1_1]
    
#calculating 1yr 2yr rates
threeyrspot=spottable.iloc[5].tolist()
forward2=[]
for i in range (0,10):
    forward1_2=(threeyrspot[i]*3-oneyrspot[i])/2
    forward2=forward2+[forward1_2]
#calculating 1yr 3yr rates
fouryrspot=spottable.iloc[7].tolist()
forward3=[]
for i in range (0,10):
    forward1_3=(fouryrspot[i]*4-oneyrspot[i])/3
    forward3=forward3+[forward1_3]
#calculating 1yr 4yr rates
fiveyrspot=spottable.iloc[9].tolist()
forward4=[]
for i in range (0,10):
    forward1_4=(fiveyrspot[i]*5-oneyrspot[i])/4
    forward4=forward4+[forward1_4]
#generate a forward rate table
forwardtable=p.DataFrame([forward1,forward2,forward3,forward4],columns=["1","2",'3','4','5','6','7','8','9','10'])
plt.subplot(1,3,3)
x=['2','3','4','5']#x axis periods left to maturity
plt.xlabel('1yr to x year forward rate', fontsize=10)
plt.ylabel('forward rate')
plt.plot(x,forwardtable['1'] ,label='Jan10')
plt.plot(x,forwardtable['2'] ,label='Jan11')
plt.plot(x,forwardtable['3'] ,label='Jan12')
plt.plot(x,forwardtable['4'],label='Jan13')
plt.plot(x,forwardtable['5'] ,label='Jan14')
plt.plot(x,forwardtable['6'],label='Jan17')
plt.plot(x,forwardtable['7'],label='Jan18')
plt.plot(x,forwardtable['8'] ,label='Jan19')
plt.plot(x,forwardtable['9'],label='Jan20')
plt.plot(x,forwardtable['10'] ,label='Jan21')
plt.title('Forward curve')
plt.legend()
plt.show()
#5  Find the vectors xij first
import math as m
i_1=ytmtable.iloc[1].tolist()
X1j=[]
for i in range (0,9):
    xij=m.log(i_1[i+1]/i_1[i])
    X1j=X1j+[xij]

i_2=ytmtable.iloc[3].tolist()
X2j=[]
for i in range (0,9):
    xij2=m.log(i_2[i+1]/i_2[i])
    X2j=X2j+[xij2]
i_3=ytmtable.iloc[5].tolist()
X3j=[]
for i in range (0,9):
    xij3=m.log(i_3[i+1]/i_3[i])
    X3j=X3j+[xij3]
i_4=ytmtable.iloc[7].tolist()
X4j=[]
for i in range (0,9):
    xij4=m.log(i_4[i+1]/i_4[i])
    X4j=X4j+[xij4]
i_5=ytmtable.iloc[9].tolist()
X5j=[]
for i in range (0,9):
    xij5=m.log(i_5[i+1]/i_5[i])
    X5j=X5j+[xij5]
vectors=p.DataFrame([X1j,X2j,X3j,X4j,X5j],columns=['1','2','3','4','5','6','7','8','9'])
vectorstranspose=p.DataFrame([vectors['1'].tolist(),vectors['2'].tolist(),vectors['3'].tolist(),vectors['4'].tolist(),vectors['5'].tolist(),vectors['6'].tolist(),vectors['7'].tolist(),vectors['8'].tolist(),vectors['9'].tolist()])
covmatrix=p.DataFrame.cov(vectorstranspose)
import scipy.linalg as la
eigenvalue=la.eig(covmatrix)
#find the covariance for daily log forward rate
i_11=forwardtable.iloc[0].tolist()
X11j=[]
for i in range (0,9):
    x1ij=m.log(i_11[i+1]/i_11[i])
    X11j=X11j+[x1ij]

i_22=forwardtable.iloc[1].tolist()
X22j=[]
for i in range (0,9):
    xij22=m.log(i_22[i+1]/i_22[i])
    X22j=X22j+[xij22]
i_33=forwardtable.iloc[2].tolist()
X33j=[]
for i in range (0,9):
    xij33=m.log(i_33[i+1]/i_33[i])
    X33j=X33j+[xij33]
i_44=forwardtable.iloc[3].tolist()
X44j=[]
for i in range (0,9):
    xij44=m.log(i_44[i+1]/i_44[i])
    X44j=X44j+[xij44]
vectors1=p.DataFrame([X11j,X22j,X33j,X44j],columns=['1','2','3','4','5','6','7','8','9'])
vectorstranspose1=p.DataFrame([vectors1['1'].tolist(),vectors1['2'].tolist(),vectors1['3'].tolist(),vectors1['4'].tolist(),vectors1['5'].tolist(),vectors1['6'].tolist(),vectors1['7'].tolist(),vectors1['8'].tolist(),vectors1['9'].tolist()])
covmatrix1=p.DataFrame.cov(vectorstranspose1)
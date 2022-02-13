import numpy as n
from scipy.misc import derivative # to be able to calculate derivative directly
import pandas as p # to be able to import data set
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
for i in range(0,10):
#one[10] is the coupon rate, and spot rate1 = (facevalue+coupon payment)/dirty price -1 
    sprate=(100*one[10]/2+100)/one[i]-1
    spotrate1=spotrate1+[sprate*2]
# find the second spot rate for 10 dates
two= data_withdirtyprice.iloc[1].tolist() # the dirty prices of the bound with 2 payment left for 10 days
spotrate2=[]
for i in range(0,10):
#Two[10] is the coupon rate and the formula for second spot rate is the following equation
    sprate2=((100+two[10]*100/2)/(two[i]-two[10]*100/2/(spotrate1[i]/2+1)))**(1/2)-1
    spotrate2=spotrate2+[sprate2*2]
# find the third spot rate for 10 dates
three= data_withdirtyprice.iloc[2].tolist() # the dirty prices of the bound with 3 payment left for 10 days
spotrate3=[]
for i in range(0,10):
    sprate3=((100+three[10]*100/2)/(three[i]-three[10]*100/2/(spotrate1[i]/2+1)-three[10]*100/2/(spotrate2[i]/2+1)**2))**(1/3)-1
    spotrate3=spotrate3+[sprate3*2]   
# find the forth spot rate for 10 dates
forth=data_withdirtyprice.iloc[3].tolist() # the dirty prices of the bound with 4 payment left for 10 days
spotrate4=[]
for i in range(0,10):
    sprate4=((100+forth[10]*100/2)/(forth[i]-forth[10]*100/2/(spotrate1[i]/2+1)-forth[10]*100/2/(spotrate2[i]/2+1)**2-forth[10]*100/2/(spotrate3[i]/2+1)**3 ))**(1/4)-1
    spotrate4=spotrate4+[sprate4*2]  
# find the fifth spot rate for 10 dates
fifth=data_withdirtyprice.iloc[4].tolist() # the dirty prices of the bound with 5 payments left for 10 days
spotrate5=[]
for i in range(0,10):
    coupon5=fifth[10]*100/2
    sprate5=((100+coupon5)/(fifth[i]-coupon5/(spotrate1[i]/2+1)-coupon5/(spotrate2[i]/2+1)**2-coupon5/(spotrate3[i]/2+1)**3-coupon5/(spotrate4[i]/2+1)**4))**(1/5)-1
    spotrate5=spotrate5+[sprate5*2] 
# find the sixth spot rate for 10 dates
sixth=data_withdirtyprice.iloc[5].tolist() # the dirty prices of the bound with 6 payments left for 10 days
spotrate6=[]
for i in range(0,10):
    coupon6=sixth[10]*100/2
    sprate6=((100+coupon6)/(sixth[i]-coupon6/(spotrate1[i]/2+1)-coupon6/(spotrate2[i]/2+1)**2-coupon6/(spotrate3[i]/2+1)**3-coupon6/(spotrate4[i]/2+1)**4-coupon6/(spotrate5[i]/2+1)**5))**(1/6)-1
    spotrate6=spotrate6+[sprate6*2] 
# find the seventh spot rate for 10 dates
seventh=data_withdirtyprice.iloc[6].tolist() # the dirty prices of the bound with 7 payments left for 10 days
spotrate7=[]
for i in range(0,10):
    coupon7=seventh[10]*100/2
    sprate7=((100+coupon7)/(seventh[i]-coupon7/(spotrate1[i]/2+1)-coupon7/(spotrate2[i]/2+1)**2-coupon7/(spotrate3[i]/2+1)**3-coupon7/(spotrate4[i]/2+1)**4-coupon7/(spotrate5[i]/2+1)**5-coupon7/(spotrate6[i]/2+1)**6))**(1/7)-1
    spotrate7=spotrate7+[sprate7*2] 
# find the eighth spot rate for 10 dates
eighth=data_withdirtyprice.iloc[7].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate8=[]
for i in range(0,10):
    coupon8=eighth[10]*100/2
    sprate8=((100+coupon8)/(eighth[i]-coupon8/(spotrate1[i]/2+1)-coupon8/(spotrate2[i]/2+1)**2-coupon8/(spotrate3[i]/2+1)**3-coupon8/(spotrate4[i]/2+1)**4-coupon8/(spotrate5[i]/2+1)**5-coupon8/(spotrate6[i]/2+1)**6-coupon8/(spotrate7[i]/2+1)**7))**(1/8)-1
    spotrate8=spotrate8+[sprate8*2] 
# find the ninth spot rate for 10 dates
ninth=data_withdirtyprice.iloc[8].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate9=[]
for i in range(0,10):
    coupon9=ninth[10]*100/2
    sprate9=((100+coupon9)/(ninth[i]-coupon9/(spotrate1[i]/2+1)-coupon9/(spotrate2[i]/2+1)**2-coupon9/(spotrate3[i]/2+1)**3-coupon9/(spotrate4[i]/2+1)**4-coupon9/(spotrate5[i]/2+1)**5-coupon9/(spotrate6[i]/2+1)**6-coupon9/(spotrate7[i]/2+1)**7-coupon9/(spotrate8[i]/2+1)**8))**(1/9)-1
    spotrate9=spotrate9+[sprate9*2] 
# find the ninth spot rate for 10 dates
tenth=data_withdirtyprice.iloc[9].tolist() # the dirty prices of the bound with 8 payments left for 10 days
spotrate10=[]
for i in range(0,10):
    coupon10=tenth[10]*100/2
    sprate10=((100+coupon10)/(tenth[i]-coupon10/(spotrate1[i]/2+1)-coupon10/(spotrate2[i]/2+1)**2-coupon10/(spotrate3[i]/2+1)**3-coupon10/(spotrate4[i]/2+1)**4-coupon10/(spotrate5[i]/2+1)**5-coupon10/(spotrate6[i]/2+1)**6-coupon10/(spotrate7[i]/2+1)**7-coupon10/(spotrate8[i]/2+1)**8-coupon10/(spotrate9[i]/2+1)**9))**(1/10)-1
    spotrate10=spotrate10+[sprate10*2]
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
#we know that (1+forwardrate)(1+1yrspotrate)=(1+2yrspotrate)^2
oneyrspot=spottable.iloc[1].tolist()
twoyrspot=spottable.iloc[3].tolist()
forward1=[]
for i in range (0,10):
    forward1_1=(1+twoyrspot[i])**2/(1+oneyrspot[i])-1
    forward1=forward1+[forward1_1]
    
#calculating 1yr 2yr rates
threeyrspot=spottable.iloc[5].tolist()
forward2=[]
for i in range (0,10):
    forward1_2=((1+threeyrspot[i])**3/(1+oneyrspot[i]))-1
    forward2=forward2+[forward1_2]
#calculating 1yr 3yr rates
fouryrspot=spottable.iloc[7].tolist()
forward3=[]
for i in range (0,10):
    forward1_3=((1+fouryrspot[i])**4/(1+oneyrspot[i]))-1
    forward3=forward3+[forward1_3]
#calculating 1yr 4yr rates
fiveyrspot=spottable.iloc[9].tolist()
forward4=[]
for i in range (0,10):
    forward1_4=((1+fiveyrspot[i])**5/(1+oneyrspot[i]))-1
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
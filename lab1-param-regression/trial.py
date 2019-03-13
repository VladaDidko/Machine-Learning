from numpy import *
import matplotlib.pyplot as plt
from scipy.stats import *
from scipy.interpolate import *


l = 10
x_learn = array([4*((i-1)/(l-1))-2 for i in range(1,l+1,1)])

print("LEARN VALUES")
y_learn = array([1/(1+25*i*i) for i in x_learn])
for x, y in zip(x_learn,y_learn):
    print(x, ' - ', y)

def lagrange(x_learn, y_learn, x):
    z = 0
    for j in range(len(y_learn)):
        p1 = 1
        p2 = 1
        for i in range(len(x_learn)):
            if i == j:
                p1 = p1 * 1
                p2 = p2 * 1
            else:
                p1 = p1 * (x - x_learn[i])
                p2 = p2 * (x_learn[j] - x_learn[i])
        z = z + y_learn[j] * p1 / p2
    return z

while True:
    print('Choose a method to launch: ')
    print('1 - Lagrange Polynomial')
    print('2 - Least Square')
    print('3 - Spline Approximation')
    method = int(input())
    if method == 1:
        dots = int(input('Input number of dots: '))
        step = (x_learn[len(x_learn) - 1] - x_learn[0]) / dots
        print('Step: ' + str(step))
        test_x = []
        test1 = -2
        i = 0
        while i < dots:
            test1 += step
            i += 1
            test_x.append(test1)
        test_y = [lagrange(x_learn, y_learn, i) for i in test_x]

        plt.plot(x_learn,y_learn,'ro', label = 'learning points')
        plt.plot(test_x,test_y,'b-', label = 'lagrange polynomial')
        plt.title("Lagrange method")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
    elif method == 2:
        dots = int(input('Input number of dots: '))
        deg = int(input('Please input the degree of model: '))
        p1 = polyfit(x_learn, y_learn, 1)
        pol = polyfit(x_learn, y_learn, deg)
        print(p1)
        print(pol)
        plt.plot(x_learn, y_learn, 'ro', label = 'learning points')
        plt.title("Least Squares")
        xp = linspace(-2, 2, dots)
        plt.plot(xp, polyval(pol, xp), 'b-')
        yfit = p1[0] * x_learn + p1[1]
        yresid = y_learn - yfit
        SSresid = sum(pow(yresid, 2))

        SStotal = len(y_learn) * var(y_learn)
        rsq = 1 - SSresid / SStotal
        print('------------')
        print('Prediction: '+ str(yfit))
        print('------------')
        print(rsq)
        print('Emp risk: '+str(SSresid))

        slope, intercept, r_value, p_value, std_err = linregress(x_learn, y_learn)
        print('Correlation coeficient in 2 pow: ' + str(pow(r_value, 2)))#The closer r is to 1 or –1, the less scattered the points are and the stronger the relationship.
        print(p_value)
        plt.legend()
        plt.show()

    elif method == 3:
        dots = int(input('Input number of dots: '))
        start= -2
        end = 2
        f = interp1d(x_learn, y_learn)
        f2 = interp1d(x_learn, y_learn, kind='cubic')
        xnew = linspace(start, end, dots, endpoint=True)
        plt.plot(x_learn, y_learn, 'ro', xnew, f(xnew), '-', xnew, f2(xnew), '--')
        plt.title('Spline Approximation')
        plt.ylim([start, end])
        plt.legend(['learning points', 'linear', 'cubic'], loc='best')
        plt.show()
    else: print('Please print number from 1 to 3 to choose the proper method to launch')


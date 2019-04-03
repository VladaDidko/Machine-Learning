from random import uniform, seed
from math import sqrt, pow, ceil
import scipy as sp
import random
import matplotlib.pyplot as plt

seed(9001) #random number generator

def points_generate(size, upper1, upper2):
    points_array = []
    for i in range(size):
        x1 = uniform(0, upper1)
        x2 = uniform(0, upper2)
        points_array.append((x1, x2))
    return points_array

def split_data(points_array):
    train_data = []
    test_data = []
    for p in points_array:
        if random.random() < 0.7:
            train_data.append(p)
        else:
            test_data.append(p)
    return train_data, test_data


def get_class(x):
    if (ceil(x[0]), ceil(x[1])) == (1,1):
        return 1
    elif (ceil(x[0]), ceil(x[1])) == (2,1):
        return 2
    elif (ceil(x[0]), ceil(x[1])) == (3,1):
        return 3
    elif (ceil(x[0]), ceil(x[1])) == (1,2):
        return 4
    elif (ceil(x[0]), ceil(x[1])) == (2,2):
        return 5
    elif (ceil(x[0]), ceil(x[1])) == (3,2):
        return 6
    elif (ceil(x[0]), ceil(x[1])) == (1,3):
        return 7
    elif (ceil(x[0]), ceil(x[1])) == (2,3):
        return 8
    elif (ceil(x[0]), ceil(x[1])) == (3,3):
        return 9
    else:
        return -1

def get_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return sqrt( pow(x2 - x1, 2) + pow(y2 - y1, 2) )

def find_nearest(x, points_array, h): #returns points that are within given h
    nearest_points = []
    help_points = []
    for p in points_array:
        d = get_distance(x, p)
        if d <= h:
            nearest_points.append([d, get_class(p)])
            help_points.append(p)
    return sorted(nearest_points)

def nearest(x, points_array, h):
    help_points = []
    for p in points_array:
        d = get_distance(x, p)
        if d <= h:
            help_points.append(p)
    return help_points


def max_index(data): #returns index of max dot
    if (len(data)) == 0:
        return -1
    m = data[0]
    m_ind = 0
    for i in range(len(data)):
        if data[i] > m:
            m_ind  = i
            m = data[i]
    return m_ind

def define_class(train_data, test_data, h, ker):
    testWithLabels = []
    for j in range(len(test_data)):
        stat = [0 for j in range(10)]
        nearest = find_nearest(test_data[j], train_data, h)
        if ker==0:
            for i in range(len(nearest)):
                stat[nearest[i][1]] += 1
        elif ker==1:
            for i in range(len(nearest)):
                stat[nearest[i][1]] += (1-nearest[i][0]/h)
        elif ker==2:
            for i in range(len(nearest)):
                stat[nearest[i][1]] += (1-(nearest[i][0]/h)**2)
        elif ker==3:
            for i in range(len(nearest)):
                stat[nearest[i][1]] += ((1-(nearest[i][0]/h)**2)**2)
        else:
            for i in range(len(nearest)):
                e=2.71828
                stat[nearest[i][1]] += e**(-2*(nearest[i][0]/h)**2)
        testWithLabels.append([test_data[j], max_index(stat)])
    return testWithLabels

def calculate_accuracy(test_data, my_test_data):
    return sum([int(test_data[i][1] == my_test_data[i][1]) for i in range(len(my_test_data))]) / float(len(my_test_data))


def train_h(train_data, test_data_labeled, test_data, maxsize, method):
    res = []
    variants = sp.linspace(0.5, maxsize, 10)
    for i in range(len(variants)):
        res.append(calculate_accuracy(test_data_labeled, define_class(train_data, test_data, i, method)))
    q = max_index(res)
    return [variants[q], res[q]]

# create data
x_given = (2.3243,0.343)
h = 0.34
maxsize = 3
points = points_generate(80,3,3)
trainData, testData = split_data(points)
test_labeled = [[testData[i],get_class(testData[i])]for i in range(len(testData))]
methods=["rectangular kernel", "triangular kernel", "square kernel", "super square kernel", "Gauss kernel"]
help_points = nearest(x_given,points,h)
print('Test point: '+str(x_given))
print('Points distances within the given H: '+str(find_nearest(x_given,points,h)))
print('Predicted class of test point: '+str(define_class(points,[x_given],h,1)))

res = []
temp = []
for i in range(5):
    tws = train_h(trainData, test_labeled, testData, maxsize, i)
    res.append([tws[0], tws[1], i])
    temp.append(tws[1])
max_index(temp)
# find the best h
h = res[max_index(temp)]
# create new array with label
pSize = res[max_index(temp)][0]
accuracy = res[max_index(temp)][1]
mNum = res[max_index(temp)][2]
res1 = define_class(trainData, testData, pSize, mNum)

print('__________________')
print("THE BEST H:", pSize)
print("THE BEST KERNEL:", methods[mNum])
print("ACCURACY:", accuracy)

plt.figure(1)
plt.axis([0, 3, 0, 3])
plt.yticks([1, 2, 3], ['1', '2', '3'])
plt.xticks([1, 2, 3], ['1', '2', '3'])
plt.scatter(*zip(*points), marker=r'.') # show all objects
plt.scatter(*zip(*help_points), marker='.',color='orange')
plt.plot(x_given[0],x_given[1],'r*') # show the object x
plt.grid()
plt.show()
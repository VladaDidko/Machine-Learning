from random import uniform, seed
import itertools
from math import sqrt, pow, floor,ceil
import matplotlib.pyplot as plt


seed(9001) #random number generator


def points_generate(size, upper1, upper2):
    points_array = []
    for i in range(size):
        x1 = uniform(0, upper1)
        x2 = uniform(0, upper2)
        points_array.append((x1, x2))
    return points_array

def get_distance(a, b): # returns distance between two points
    x1, y1 = a
    x2, y2 = b
    return sqrt( pow(x2 - x1, 2) + pow(y2 - y1, 2) ) #euclidean metrics

def get_class(x):
    return (ceil(x[0]), ceil(x[1]))

def getting(x):
    if get_class(x) == (1,1):
        return 1
    elif get_class(x) == (2,1):
        return 2
    elif get_class(x) == (3,1):
        return 3
    elif get_class(x) == (1,2):
        return 4
    elif get_class(x) == (2,2):
        return 5
    elif get_class(x) == (3,2):
        return 6
    elif get_class(x) == (1,3):
        return 7
    elif get_class(x) == (2,3):
        return 8
    elif get_class(x) == (3,3):
        return 9
    else:
        return 0


def knn_classification(x, points_array, k, with_graphics=True): # returns supposed class by closest neighbors
    closest = sorted(points_array, key=lambda o: get_distance(x, o), reverse=False)[:k]

    if with_graphics:
        plt.figure(1)
        plt.subplot(221)
        plt.axis([0, 3, 0, 3])
        plt.yticks([1,2,3],['1','2','3'])
        plt.scatter(*zip(*points_array), marker=r'.') # show all objects
        plt.plot(x[0],x[1],'r*') # show the object x
        plt.grid()

        plt.subplot(222)
        plt.axis([0, 3, 0, 3])
        plt.yticks([1, 2, 3], ['1', '2', '3'])
        plt.scatter(*zip(*points_array), marker=r'.')
        # plt.scatter(*zip(*etalons),marker = '*', color = 'orange')
        plt.scatter(*zip(*closest), marker='.',color='black') # show only closest objects
        plt.plot(x[0],x[1],'r*') # show the object x
        plt.grid()
        # plt.show()

    classDict = {}
    for item in closest:
        cls = get_class(item)
        classDict[cls] = classDict.get(cls,0)+1
    maxCls,maxCount = 0,0
    for(cls,val) in classDict.items():
        if maxCount < val:
            maxCount = val
            maxCls = cls
    return  maxCls


def loo(points_array, k):
    num = 0
    for p in points_array:
        p = points_array.pop()
        actual = get_class(p)
        predicted = knn_classification(p,points_array,k,with_graphics=False)
        if actual == predicted:
            num+=1
        points_array.insert(0,p)
    return num

def dist_to_all(x, xs):
    d = sum([get_distance(x, xs[i]) for i in range(len(xs))])
    return d

def find_etalon(points_array,x):
    classes = [{p:getting(p)} for p in points_array]
    keyfunc = lambda d:next(iter(d.values()))
    classes_sorted = {k: [x for d in g for x in d] for k, g in itertools.groupby(sorted(classes,key=keyfunc),key = keyfunc)}

    etalons = []
    for key,value in classes_sorted.items():
        distances = {v:dist_to_all(v,value) for v in value}
        etalons.append(min(distances, key=distances.get))
    plt.scatter(*zip(*etalons),marker='*',color='orange')
    etalons_dist = {e:get_distance(e,x) for e in etalons}
    etalons_dist_sorted = sorted(etalons_dist,key=etalons_dist.get)
    etalon = etalons_dist_sorted.pop(0)
    result = get_class(etalon)
    return result

points = points_generate(120, 3, 3)
data = [[p,getting(p)]for p in points]
print(points)
x_given = (1.45343, 2.16748)
print('ACTUAL CLASS: '+str(get_class(x_given)))
print('PREDICTED CLASS: ', knn_classification(x_given, points, k=4))
hits = []
for k in range(1,20):
    hits.append((k, loo(points,k)))
print(hits)
print('ETALON METHOD. POINT CLASS IS: '+str(find_etalon(points,x_given)))
plt.show()

import numpy as np
import scipy.stats as stats 

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    # Test 1
    np.random.seed(0)
    x = list(range(0,100))
    y = [value * 2 + np.random.normal(0,25) for value in x]
    
    mslr = MySimpleLinearRegressor()
    mslr.fit(x,y)
    sklearn_line = stats.linregress(x, y)
    assert np.allclose(mslr.slope, sklearn_line.slope)
    assert np.allclose(mslr.intercept, sklearn_line.intercept)

    # Test 2
    np.random.seed(10)
    x2 = list(range(0,100))
    slope = np.random.normal(20,88)
    y2 = [value2 * slope + np.random.normal(0,25) for value2 in x2]
    
    mslr.fit(x2,y2)
    sklearn_line2 = stats.linregress(x2, y2)
    assert np.allclose(mslr.slope, sklearn_line2.slope)
    assert np.allclose(mslr.intercept, sklearn_line2.intercept)

def test_simple_linear_regressor_predict():
    mslr = MySimpleLinearRegressor()
    
    # Test 1, simple slope of 3
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 3, 6, 9, 12, 15]
    mslr.fit(x,y)
    X_test = [[6],[7]]
    y_predicted = mslr.predict(X_test)
    assert(y_predicted == [18, 21]) 

    # Test 2, float slope test 
    x2 = [12, 45, 67, 34, 2]
    y2 = [3, 11, 23, 9, .6]
    mslr.fit(x2,y2)
    X_test2 = [[85],[44]]
    y_predicted2 = mslr.predict(X_test2)
    #print("y_predict:", y_predicted2)
    assert(y_predicted2 == [26.764032616753152, 13.269592290585619]) 

def test_kneighbors_classifier_kneighbors():
    mknc = MyKNeighborsClassifier()
    #print()
    # Test 1, 4 instance training set example traced in class on the iPad
    x = [
        [7,7], 
        [7,4], 
        [3,4], 
        [1,4]
    ]
    y = ["bad", "bad", "good", "good"]
    mknc.fit(x,y)
    test = [[3,7]]
    distances, indices = mknc.kneighbors(test)
    #print("distances: ", distances)
    #print("indices: ", indices)
    assert(distances == [[0.6666666666666667, 1.0, 1.0540925533894598]])
    assert(indices == [[0, 2, 3]])

    # Test 2, Use the 8 instance training set example from ClassificationFun/main.py
    x2 = [
        [3,2],
        [6,6],
        [4,1],
        [4,4],
        [1,2],
        [2,0],
        [0,3],
        [1,6]
    ]
    y2 =["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    mknc.fit(x2,y2)
    test2 = [[2, 3]]
    distances2, indices2 = mknc.kneighbors(test2)
    #print("distances2: ", distances2)
    #print("indices2: ", indices2)
    assert(distances2 == [[0.23570226039551584, 0.23570226039551587, 0.3333333333333333]])
    assert(indices2 == [[4, 0, 6]])

    # Test 3, Use Bramer 3.6 Self-assessment exercise 2
    x3 = [
        [.8,6.3],
        [1.4,8.1],
        [2.1,7.4],
        [2.6,14.3],
        [6.8,12.6],
        [8.8,9.8],
        [9.2,11.6],
        [10.8,9.6],
        [11.8,9.9],
        [12.4,6.5],
        [12.8,1.1],
        [14.0,19.9],
        [14.2,18.5],
        [15.6,17.4],
        [15.8,12.2],
        [16.6,6.7],
        [17.4,4.5],
        [18.2,6.9],
        [19.0,3.4],
        [19.6,11.1]
    ]
    y3 = ['-','-','-','+','-','+','-','+','+','+','-','-','-','-','-','+','+','+','-','+']
    mknc.__init__(5)
    mknc.fit(x3,y3)
    test3 = [[9.1, 11.0]]
    distances3, indices3 = mknc.kneighbors(test3)
    #print("distances3: ", distances3)
    #print("indices3: ", indices3)
    assert(distances3 == [[0.032355119842011795, 0.06579423870666472, 0.1171421039656662, 0.14903112474597766, 0.15507850784163038]])
    assert(indices3 == [[6, 5, 7, 4, 8]])

def test_kneighbors_classifier_predict():
    mknc = MyKNeighborsClassifier()
    #print()
    # Test 1, 4 instance training set example traced in class on the iPad
    x = [
        [7,7], 
        [7,4], 
        [3,4], 
        [1,4]
    ]
    y = ["bad", "bad", "good", "good"]
    mknc.fit(x,y)
    test = [[3,7]]
    y_predicted = mknc.predict(test)
    #print("y_predicted: ", y_predicted)
    assert(y_predicted == ['good'])

    # Test 2, Use the 8 instance training set example from ClassificationFun/main.py
    x2 = [
        [3,2],
        [6,6],
        [4,1],
        [4,4],
        [1,2],
        [2,0],
        [0,3],
        [1,6]
    ]
    y2 =["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    mknc.fit(x2,y2)
    test2 = [[2, 3]]
    y_predicted2 = mknc.predict(test2)
    #print("y_predicted2: ", y_predicted2)
    assert(y_predicted2 == ['no'])
    
    # Test 3, Use Bramer 3.6 Self-assessment exercise 2
    x3 = [
        [.8,6.3],
        [1.4,8.1],
        [2.1,7.4],
        [2.6,14.3],
        [6.8,12.6],
        [8.8,9.8],
        [9.2,11.6],
        [10.8,9.6],
        [11.8,9.9],
        [12.4,6.5],
        [12.8,1.1],
        [14.0,19.9],
        [14.2,18.5],
        [15.6,17.4],
        [15.8,12.2],
        [16.6,6.7],
        [17.4,4.5],
        [18.2,6.9],
        [19.0,3.4],
        [19.6,11.1]
    ]
    y3 = ['-','-','-','+','-','+','-','+','+','+','-','-','-','-','-','+','+','+','-','+']
    mknc.__init__(5)
    mknc.fit(x3,y3)
    test3 = [[9.1, 11.0]]
    y_predicted3 = mknc.predict(test3)
    #print("y_predicted3: ", y_predicted3)
    assert(y_predicted3 == ['+'])
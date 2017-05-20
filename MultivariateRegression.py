import numpy as np 
import math

x_data = np.loadtxt('ex3x.dat')
y_data = np.loadtxt('ex3y.dat')

# FOR TESTING PURPOSES ONLY
'''print("X Feature Vector Data: ")
for xElement in myXData:
	print(xElement[0])
	print(xElement[1])
	print()
print("")
print("Y Vector Data: ")
for yElement in myYData:
	print(yElement)
print("")'''
# FOR TESTING PURPOSES ONLY

# Can implement Gradient Descent Algorithm, Normal Equations Method, or both

# (1) Normal Equation Method

# (Step 1): Construct the Design Matrix (X)
x_count = 0
X = 0

for x_element in x_data:
	if x_count == 0:
		X = np.array([[1, x_element[0], x_element[1]]])
	else:
		X = np.append(X, [[1, x_element[0], x_element[1]]], axis=0)
	x_count+=1

# (Step 2): Construct the y-value matrix
Y = 0
y_count = 0

for y_element in y_data:
	if y_count == 0:
		Y = np.array([[y_element]])
	else:
		Y = np.append(Y, [[y_element]], axis=0)
	y_count+=1

# (Step 3): Compute Theta = (X^{T}X)^{-1} (X^{T}Y)
X_T = np.transpose(X)
X_T_X = np.dot(X_T, X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_T_Y = np.dot(X_T, Y)

theta = np.dot(X_T_X_inv, X_T_Y)

print("Resulting Theta Value: " + str(np.dot(X_T_X_inv, X_T_Y)))

#(Step 4): Compute the predicted y-value for a given feature vector using (theta)^{T}(feature vector)
theta_T = np.transpose(theta)
feature_vector = np.array([1, 1650, 3])
prediction = np.dot(theta_T, feature_vector)

print("Predicted price for 1650 square foot house with 3 bedrooms: $" + str(prediction[0]))

# (2) Gradient Descent Method

print("GRADIENT DESCENT METHOD")

thetas = np.array([0,0,0])
learning_rate = 1.0

x1_average = 0
x2_average = 0

x1_min = 10000
x1_max = 0

x2_min = 10000
x2_max = 0

for x_element in x_data:
	x1_average += x_element[0]
	x2_average += x_element[1]

	if x_element[0] < x1_min:
		x1_min = x_element[0]
	if x_element[0] > x1_max:
		x1_max = x_element[0]
	if x_element[1] < x2_min:
		x2_min = x_element[1]
	if x_element[1] > x2_max:
		x2_max = x_element[1]

x1_average /= 47
x2_average /= 47

x1_range = x1_max-x1_min
x2_range = x2_max-x2_min


# INPUTS:
# (1) Specific row of X matrix (constructed above)
# (2) Current theta values (theta_{0}, theta_{1}, theta_{2})

# OUTPUTS:
# H_{theta}(x^(i)) = (theta_0) + (theta_1)*(x^(i)_{1}) + (theta_2)*(x^(i)_{2})
def hypothesis(x_i, thetas):
	return thetas[0] + thetas[1]*((x_i[0]-x1_average)/x1_range) + thetas[2]*((x_i[1]-x2_average)/x2_average)

# INPUTS:
# (1) y-vector

# OUTPUTS:
# 1/m * Sum(i = 1...m) [hypothesis(x_{i}, thetas) - y_{i}]
def derivative_theta_0(y):
	curr_sum = 0
	curr_index = 0
	for x_element in x_data:
		curr_sum += (hypothesis(x_element, thetas)-y[curr_index,0:])
		curr_index+=1
	return (curr_sum/47)

# INPUTS:
# (1) y-vector

# OUTPUTS:
# 1/m * Sum(i=1...m) [hypothesis(x_{i}, thetas) - y_{i}] * x_{i}_{1}
def derivative_theta_1(y):
	curr_sum = 0
	curr_index = 0
	for x_element in x_data:
		curr_sum += ( (hypothesis(x_element, thetas)-y[curr_index,0:]) * (x_element[0]-x1_average)/x1_range )
		curr_index+=1
	return (curr_sum/47)

# INPUTS:
# (1) y-vector

# OUTPUTS:
# 1/m * Sum(i=1...m) [hypothesis(x_{i}, thetas) - y_{i}] * x_{i}_{2}
def derivative_theta_2(y):
	curr_sum = 0
	curr_index = 0
	for x_element in x_data:
		curr_sum += ( (hypothesis(x_element, thetas)-y[curr_index,0:]) * (x_element[1]-x2_average)/x2_range )
		curr_index+=1
	return (curr_sum/47)


num_iterations = 0
'''while math.fabs(derivative_theta_0(Y)[0]) > 0.1 and math.fabs(derivative_theta_1(Y)[0])>0.1 and math.fabs(derivative_theta_2(Y)[0])>0.1:
	thetas[0] = thetas[0] - learning_rate * derivative_theta_0(Y)[0]
	thetas[1] = thetas[1] - learning_rate * derivative_theta_1(Y)[0]
	thetas[2] = thetas[2] - learning_rate * derivative_theta_2(Y)[0]
	print("new theta value: " + str(thetas))
	num_iterations+=1
'''
print()
print("Theta Vector Values: [" + str(thetas[0]) + ", " + str(thetas[1]) + ", " + str(thetas[2]) + "]")





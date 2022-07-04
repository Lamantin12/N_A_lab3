from numpy import zeros, copy, dot, inf
from numpy.linalg import norm

##############################################################################################################################################

def solution_task_2(matrix_, vector_):
    result = zeros((matrix_.shape[0],1))
    matrix = copy(matrix_)
    vector = copy(vector_)
    for i in range(matrix.shape[0]):
        if (i-6) < 0 :
            result[i] = vector[i] / matrix[i,i]
        elif (i-15) < 0:
            result[i] = vector[i] / matrix[i,i] - matrix[i,i-6] / matrix[i,i] * result[i-6]
        else:
            result[i] = vector[i] / matrix[i,i] - matrix[i,i-6] / matrix[i,i] * result[i-6] - matrix[i,i-16] / matrix[i,i] * result[i-16]
    return result


##############################################################################################################################################

def method_of_minimal_residual(matrix, vector, epsilon):
    x = zeros(vector.shape[0])
    r = dot(matrix, x) - vector
    r0 = r
    tau = dot(dot(matrix, r), r) / dot(dot(matrix, r), dot(matrix, r))
    x = x - tau*r
    while (norm(r, inf)/norm(r0, inf)) > epsilon:
        r = dot(matrix, x) - vector
        tau = dot(dot(matrix, r), r) / dot(dot(matrix, r), dot(matrix, r))
        x = x - tau*r
    return x

##############################################################################################################################################
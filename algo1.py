# -*- coding:utf-8 *-*
import numpy as np
import helpers
import kernel_helpers as k_helpers
from sklearn import svm


#判断矩阵x是不是所有特征值都大于0，即判断矩阵x是否为正定，特征值均大于0的是正定矩阵，特征值均为非负的，则矩阵是半正定的
def is_pos_def(x):
    #判断x是否为正定矩阵
    return np.all(np.linalg.eigvals(x) > 0) #np.linalg.eigvals(x)函数计算的是矩阵x的特征值
                                            #np.linalg.eig(x)函数返回的是矩阵的特征值和特征向量  一个list>0,得到的是一个数组，要么为false、要么为true
                                            #np.all(a)函数，若a全为true则返回true，否则返回false

def find_kernel_weights(X, y, kernel_functions):
    # The number of kernels
    M = len(kernel_functions)

    #The number of examples
    n = len(y)

    #The weights of each kernel
    #Initialized to 1/M
    d = np.ones(M) / M   #每个核的权重为1/M
    D = np.ones(M)       #全部初始化为1

    #Just a placeholder for something that gets updated later
    dJ = '-20'

    #Stores all the individual kernel matrices
    kernel_matrices = k_helpers.get_all_kernels(X, kernel_functions)   #获取不同核函数的核矩阵

    #Creates y matrix for use in SVM later
    #y's should be -1, 1 before doing this.
    y_mat = np.outer(y, y)   #由于svm的目标函数里有yi*yj，故提前计算好y相乘的矩阵，为后面做准备

    #Gets constraints for running SVM
    box_constraints = helpers.get_box_constraints(n, 1.0)  #每个样本给出一个取值限制区间[0,c=1.0],list长为样本个数，每个元素为一个两个值的list

    #Gets starting value for SVM
    alpha0 = np.zeros(n)   #初始化n个alpha全为0

    iteration = 0

    #Loops until stopping criterion reached
    while (max(D) != 0 and not helpers.stopping_criterion(dJ, d, 0.01)):
        iteration += 1
        print "iteration and weights:", iteration, d

        combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d)
        combined_kernel_func = k_helpers.get_combined_kernel_function(kernel_functions, d)

        #Gets J, also calculates the optimal values for alpha
        alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha0, box_constraints)
        J *= -1

        #Gradient of J w.r.t d (weights)
        dJ = helpers.compute_dJ(kernel_matrices, y_mat, alpha)
        
        #The index of the largest component of d
        mu = d.argmax()

        #Descent direction
        #Basically, we are calculating -1 * reduced gradient of J w.r.t d,
        #using the index of the largest component of d as our "mu"
        #in the reduced gradient calculation
        D = helpers.compute_descent_direction(d, dJ, mu)
        D = helpers.fix_precision_of_vector(D, 0)
        J_cross = 0
        d_cross = d.copy()
        D_cross = D.copy()

        sub_iteration = 0

        #Get maximum admissible step size in direction D
        while (J_cross < J):
            sub_iteration += 1
            d = d_cross.copy()
            D = D_cross.copy()

            print '  J:', J, '| J_cross:', J_cross
            print '    d cross', d_cross
            print '    d cross sum', sum(d_cross)

            print '    D cross', D_cross
            print '    D cross sum', sum(D_cross)

            combined_kernel_matrix = k_helpers.get_combined_kernel(kernel_matrices, d)
            alpha, J, info = helpers.compute_J(combined_kernel_matrix, y_mat, alpha, box_constraints)
            J *= -1

            #Maximum admissible step size
            gamma_max = 123456

            #argmax of above
            v = -0.123456

            #Find gamma_max and v
            for m in range(M):
                if D[m] < 0:
                    d_D_quotient = -1 * d[m] / D[m]
                    if d_D_quotient < gamma_max:
                        gamma_max = d_D_quotient
                        v = m

            d_cross = d + gamma_max * D

            #Not strictly necessary, but helps avoid precision errors
            d_cross[v] = 0


            D_cross[mu] = D[mu] + D[v]
            D_cross[v] = 0

            d_cross = helpers.fix_precision_of_vector(d_cross, 1)
            D_cross = helpers.fix_precision_of_vector(D_cross, 0)

            combined_kernel_matrix_cross = k_helpers.get_combined_kernel(kernel_matrices, d_cross)
            alpha_cross, J_cross, cross_info = helpers.compute_J(combined_kernel_matrix_cross, y_mat, alpha, box_constraints)
            J_cross *= -1
            print '    new J cross', J_cross

        #Line search along D for gamma (step) in [0, gamma_max]
        # gamma = helpers.get_armijos_step_size()
        gamma = helpers.get_armijos_step_size(kernel_matrices, d, y_mat, alpha,
                                              box_constraints, gamma_max, J_cross,
                                              D, dJ)
        print "I want to see d", d
        print 'gamma:', gamma
        print 'D:', D
        d += gamma * D
        d = helpers.fix_precision_of_vector(d, 1)
        print "I want to see d",d

    #Return final weights

    return d

# X = np.ones((5,5)) + (np.eye(5) * 2)
# X[2, 1] = 3
# X[3, 1] = 2.5
# y = np.ones((5,1))
# y[2] = -1
# y[3] = -1
# kernel_functions = [k_helpers.create_poly_kernel(degree=3, gamma=0.25),
#                     k_helpers.linear_kernel,
#                     k_helpers.create_poly_kernel(degree=5, gamma=0.3)]

# d = find_kernel_weights(X, y, kernel_functions)

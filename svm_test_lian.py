# -*- coding:utf-8 *-*
# @Time    : 2018/11/14 0014 17:53
# @Author  : LQY
# @File    : svm_test_lian.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from pre_data import shuffle_data
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit,GridSearchCV
from sklearn.feature_selection import SelectKBest,f_classif
import evaluate as evl
from sklearn.svm import SVC
import pre_data as pre
from feature_create import get_Chars,creat_frequency_mat
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import time


def select_feature(data_x, label, k_features):
    selector = SelectKBest(f_classif, k_features)  # 6700)
    selector.fit(data_x, label)
    x = selector.transform(data_x)
    return x, selector


# if __name__ == '__main__':
#     w_range = np.arange(0.05, 0.75, 0.05)
#     auc_now = 0.0
#     w_now = 0.05
#     for w in w_range:
#
#         #path='%s%.2f%s' % ('./AntiPro_Feature_Pseaac/v2_lambda5_w',w,'__feature.csv')
#         path = '%s%.2f%s' % ('./stash/v2_lambda5_w', w, '__feature.csv')
#         print("w=",w)
#         data = pd.read_csv(path, index_col=0, header=0)
#
#         data = shuffle_data(data)
#         label_t=data['label']
#         #print(label_t)
#         del(data['label'])
#         X=data.values
#         #print(X)
#
#
#         # SVC参数调优
#         # Cancerlectin
#         C_range = np.logspace(30, -5, 5, base=2)
#         gamma_range = np.logspace(10, -40, 5, base=2)
#         # # Bacteriophage
#         # C_range = np.logspace(15, 5, 11, base=2)
#         # gamma_range = np.logspace(-15, -25, 11, base=2)
#         param_grid = dict(gamma=gamma_range, C=C_range)
#         cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
#         grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
#         grid.fit(X, label_t)
#         print("The best C is 2^%d, best gamma is 2^%d, with a score of %0.2f"
#               % (np.log2(grid.best_params_['C']), np.log2(grid.best_params_['gamma']),
#                  grid.best_score_))
#         clf = SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'],
#                   probability=True)
#
#         # 训练集预测评价
#
#         skf = StratifiedKFold(n_splits=10)
#         print(len(label_t))
#         pred_label = np.ones(len(label_t)) * (-1)
#         pred_proba = np.ones((len(label_t), 2)) * (-1)
#         print(X.shape[0])
#         for idx, (train_idx, valid_idx) in enumerate(skf.split(X, label_t)):
#             print('The %dth fold:--------------------' % (idx + 1))
#             train_x = X[train_idx]
#             train_y = label_t[train_idx]
#             valid_x = X[valid_idx]
#             valid_y = label_t[valid_idx]
#
#             clf.fit(train_x, train_y)
#             pred_label[valid_idx] = clf.predict(valid_x)
#             pred_proba[valid_idx] = clf.predict_proba(valid_x)
#
#         pred_label = pred_label.astype(int)
#
#         print("10-CV training result：")
#         auc_value = evl.evaluate(label_t, pred_label, pred_proba)
#         if auc_now < auc_value:
#             auc_now = auc_value
#             w_now = w
#         print('Fnished!!!!!!!!!!')
#     print("at max_auc:auc=", auc_now, "   w=", w_now)


# if __name__ == '__main__':
#
#     # path='./Features/Protein/Antioxidant/3-gap_ant.csv'
#     # data = pd.read_csv(path, index_col=0, header=0)
#     p_path = './RawData/Protein/Thermophilic/thermophilic.txt'
#     n_path = './RawData/Protein/Thermophilic/non-thermophilic.txt'
#     # gap2_fea_path = './Features/Protein/Thermophilic/2_gap.csv'
#     # gap2fea_mat = pd.read_csv(gap2_fea_path, index_col=0, header=0)
#     # del gap2fea_mat['label']
#     data, label = pre.load_p_n_file(p_path, n_path, shuffle=False)
#     Chars = get_Chars(data)
#     fre_mat = creat_frequency_mat(Chars, data)
#     fre_mat['label']=label
#     print(fre_mat)
#     fre_mat.to_csv('./Features/Protein/Thermophilic/amino_acid_fre_ant.csv')
#
#     #fre_mat = pd.concat([fre_mat, gap2fea_mat], axis=1)
#
#     for i in range(1,fre_mat.shape[1]):
#         auc_re_list=[]
#         fre_mat, selector = select_feature(fre_mat, label, i)
#         fre_mat=pd.DataFrame(fre_mat)
#
#         fre_mat['label']=label
#         print(type(fre_mat))
#         #print(selector)
#         data=fre_mat
#         data = shuffle_data(data)
#         label_t=data['label']
#         print(data.shape)
#         #print(label_t)
#         del(data['label'])
#         X=data.values
#         #print(X)
#         #print(X.shape)
#         #X=data
#
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)
#
#
#         # SVC参数调优
#         # Cancerlectin
#
#
#         # 训练集预测评价
#
#         skf = StratifiedKFold(n_splits=10)
#         print(len(label_t))
#         pred_label = np.ones(len(label_t)) * (-1)
#         pred_proba = np.ones((len(label_t), 2)) * (-1)
#         print(X.shape[0])
#         for idx, (train_idx, valid_idx) in enumerate(skf.split(X, label_t)):
#             print('The %dth fold:--------------------' % (idx + 1))
#             train_x = X[train_idx]
#             train_y = label_t[train_idx]
#             valid_x = X[valid_idx]
#             valid_y = label_t[valid_idx]
#
#             C_range = np.logspace(10, -10, 21, base=2)
#             gamma_range = np.logspace(10, -10, 21, base=2)
#             # # Bacteriophage
#             # C_range = np.logspace(15, 5, 11, base=2)
#             # gamma_range = np.logspace(-15, -25, 11, base=2)
#             param_grid = dict(gamma=gamma_range, C=C_range)
#             cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
#             grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
#             grid.fit(train_x, train_y)
#             print("The best C is 2^%d, best gamma is 2^%d, with a score of %0.2f"
#                   % (np.log2(grid.best_params_['C']), np.log2(grid.best_params_['gamma']),
#                      grid.best_score_))
#
#
#             pred_label[valid_idx] = grid.predict(valid_x)
#             pred_proba[valid_idx] = grid.predict_proba(valid_x)
#
#         pred_label = pred_label.astype(int)
#
#         print("10-CV training result：")
#         auc_value = evl.evaluate(label_t, pred_label, pred_proba)
#         auc_re_list.append(auc_value)
#         print('Fnished!!!!!!!!!!')
#     auc_re_list=pd.DataFrame(auc_re_list)
#     auc_re_list.to_csv()
#     print(max(auc_re_list),auc_re_list.index(max(auc_re_list)))
#






if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    path_1 = r'./Features/Protein/Thermophilic/amino_acid_fre_ant.csv'
    path_2 = r'./Features/Protein/Thermophilic/0_gap.csv'
    path_3 = r'./Features/Protein/Thermophilic/1_gap.csv'
    path_4 = r'./Features/Protein/Thermophilic/2_gap.csv'

    

    file1 = open('./selected_feature.txt', 'a')
    file1.write(path_1)
    train_data1=pd.read_csv(path_1,header=0,index_col=0)
    del train_data1['label']
    train_data2 = pd.read_csv(path_2, header=0, index_col=0)
    del train_data2['label']
    train_data3 = pd.read_csv(path_3, header=0, index_col=0)
    train_data=pd.concat([train_data1,train_data2,train_data3],axis=1)
    col_list=train_data.columns
    #train_data = pre.mix_features(path_1, path_2, path_3,path_4)
    # train_data = pre.shuffle_data(train_data)
    X, y = pre.sep_x_y(train_data)
    # Create the RFE object and compute a cross-validated score.

    #
    # print("Optimal number of features : %d" % rfecv.n_features_)
    #



    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()


    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    label_t=y


    svc = SVC(kernel='linear')
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5), min_features_to_select=1,
                  scoring='accuracy')
    selector = rfecv.fit(X, label_t)
    X = selector.transform(X)
    select_colname = []
    selector.n_features_
    print(selector.get_support())
    print(selector.ranking_)
    print(sum(selector.support_))
    for i in selector.support_:
        if i:
            select_colname.append(col_list[i])
    for s in select_colname:
        file1.write(str(s))
        file1.write('\n')

    #print(selector.ranking_)


    # SVC参数调优
    # Cancerlectin


    # 训练集预测评价

    skf = StratifiedKFold(n_splits=10)
    print(len(label_t))
    pred_label = np.ones(len(label_t)) * (-1)
    pred_proba = np.ones((len(label_t), 2)) * (-1)
    print(X.shape[0])
    for idx, (train_idx, valid_idx) in enumerate(skf.split(X, label_t)):
        print('The %dth fold:--------------------' % (idx + 1))
        train_x = X[train_idx]
        train_y = label_t[train_idx]
        valid_x = X[valid_idx]
        valid_y = label_t[valid_idx]

        C_range = np.logspace(10, -10, 21, base=2)
        gamma_range = np.logspace(10, -10, 21, base=2)
        # # Bacteriophage
        # C_range = np.logspace(15, 5, 11, base=2)
        # gamma_range = np.logspace(-15, -25, 11, base=2)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
        grid = GridSearchCV(SVC(probability=True), param_grid=param_grid, cv=cv)
        grid.fit(train_x, train_y)
        print("The best C is 2^%d, best gamma is 2^%d, with a score of %0.2f"
              % (np.log2(grid.best_params_['C']), np.log2(grid.best_params_['gamma']),
                 grid.best_score_))
        pred_label[valid_idx] = grid.predict(valid_x)
        pred_proba[valid_idx] = grid.predict_proba(valid_x)

    pred_label = pred_label.astype(int)

    print("10-CV training result：")
    auc_value = evl.evaluate(label_t, pred_label, pred_proba)
    print('Fnished!!!!!!!!!!')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))



# #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# # Create SVM classification object
# svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
# model = svm.svc(kernel='linear', c=1, gamma=1)
# # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
# model.fit(X, y)
# model.score(X, y)
# #Predict Output
# predicted= model.predict(x_test)


### Import global libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

### sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

import joblib
import argparse




# Configuration parameters from command line
parser = argparse.ArgumentParser(description='Poland samples project classifiers', \
                                    epilog='Configure your experiment from the command line')
parser.add_argument('-df', '--data-file', required=True, type=str, default='./', \
    help='File with the surface parameters')
parser.add_argument('-n', '--name', required=False, type=str, default='experiment', \
    help='Name of the experiment')
parser.add_argument('-ic', '--input-channel', required=False, type=str, default='Input1', \
    help='AFM channel used for classification')
parser.add_argument('-e', '--experiment', required=True, type=str, \
    choices=['control', 'treated', 'comb1','comb2','comb3','comb4'], \
    help='Experiment to run')
parser.add_argument('-t', '--test-size', required=False, type=float, default=0.3, \
    help='Fraction of dataset to use for testing')
parser.add_argument('-r', '--rank', required=False, type=int, default=10, \
    help='Number of features after Gini reduction')
parser.add_argument('-tp', '--type', required=False, type=str, default='one_acc', \
    help='Type of the experiment. one_acc - just one accuracy computation. acc_vs_nfeat - accuracy wrt to the number of features.')
parser.add_argument('-v', '--verbose', required=False, type=str, default='', \
    help='verbosity level. If set to "high" will print all aucs or accs for further statistical significance analysis')
parser.add_argument('-mf', '--manual_features', required=False, type=str, default='', \
    help='option to input features manually bypassing GINI search. "manual" to activate')
args = parser.parse_args()
n_cpu_cores = max(os.cpu_count() - 1, 1)
np.random.seed(0)



#chans = ['Input1','Input2','Input3','Adhesion']
chans = ['Input1','Input2','Input3']

df = pd.read_excel(args.data_file,sheet_name=chans,index_col=[0])


if args.input_channel == "combined":
  print('Working with combined data.')
  df_temp = pd.DataFrame([])
  for chan in chans:
    temp = df[chan]
    temp = temp.drop(['smpl'],axis=1)
    cols = temp.columns.to_list()
    cols_upd = [chan + '_' + col for col in cols]
    temp.columns = cols_upd
    df_temp = pd.concat([df_temp,temp],axis=1)
  df = pd.concat([df_temp,df[chans[0]]['smpl']],axis=1) # add the labels back
else:	
  df = df[args.input_channel]
  if args.manual_features == 'manual':
    #print("Only using surface independent params") ### for heatmaps. because we change scale there
    #df = df[['S_a','S_q','S_sk','S_ku','S_mean','S_sc','S_dq','S_dq6','S_bi','S_dc5-10','S_dc10-50','S_dc50-95','S_dc50-100','S_ds','S_fd','smpl']]
    print("Using top parameters for ctrl and trated")
    ## df = df[['S_dc5-10', 'S_cl37', 'S_cl20', 'S_hw', 'S_tr20', 'S_dc10-50', 'S_pk', 'S_v', 'S_fd', 'S_dq6','S_tr37', 'S_dc0-5',  'S_ku', 'S_bi', 'S_z', 'S_10z', 'smpl']]   
    #df = df[['S_dc5-10', 'S_pk', 'S_tr37', 'S_cl37', 'S_tr20', 'S_dc0-5', 'S_cl20', 'S_bi', 'S_dc10-50', 'S_ku', 'S_hw', 'S_fd', 'S_z', 'S_v', 'S_sk', 'S_dq6', 'smpl']]
    ## treated/control
    #df = df[['S_pk', 'S_dc5-10', 'S_tr37', 'S_cl37', 'S_dc0-5', 'S_tr20', 'S_bi', 'S_cl20', 'S_ku', 'S_dc10-50', 'S_hw', 'S_z', 'S_fd', 'S_sk', 'S_v', 'S_dq6', 'smpl']]
    # control/treated
    ###27
    #df = df[['S_bi', 'S_sc', 'S_pk', 'S_q', 'S_dc0-5', 'S_dc5-10', 'S_mean', 'S_p', 'S_ku', 'S_z', 'S_dq6', 'S_a', 'S_dq', 'S_v', 'smpl']]
    ## treated/control
    #df = df[['S_sc', 'S_bi', 'S_q', 'S_pk', 'S_dc0-5', 'S_mean', 'S_dc5-10', 'S_p', 'S_ku', 'S_dq6', 'S_z', 'S_dq', 'S_a', 'S_v','smpl']]
    ## control/treated
    df = df[['S_tdi', 'smpl']]



if args.experiment == 'treated':
  df_hep = df.loc[df.smpl.str.contains("HEPAR|hep")]
  df_hep['lbl'] = 0
  df_hep.loc[df_hep.smpl.str.contains("TCCSUP"),'lbl']=1
  df = df_hep
  df = df.drop(index=np.random.choice(df.loc[df.lbl==1].index.to_numpy(),20,replace=False)) ## we have too many HEPAR TCCSUP samples here
elif args.experiment == 'control':
  df_ctrl = df.loc[df.smpl.str.contains("CTRL|ctrl")]
  df_ctrl['lbl'] = 0
  df_ctrl.loc[df_ctrl.smpl.str.contains("TCCSUP"),'lbl']=1
  df = df_ctrl
elif args.experiment == 'comb1': ## TCCSUP_hepar vs TCCSUP_ctrl
  df_t = df.loc[df.smpl.str.contains("TCCSUP")]
  df_t['lbl'] = 0
  df_t.loc[df_t.smpl.str.contains("HEPAR|hep"),'lbl'] = 1
  df = df_t
  df = df.drop(index=np.random.choice(df.loc[df.lbl==1].index.to_numpy(),20,replace=False)) ## we have too many HEPAR TCCSUP samples here
elif args.experiment == 'comb2': ## HCV_hepar vs HCV_ctrl
  df_t = df.loc[df.smpl.str.contains("HCV")]
  df_t['lbl'] = 0
  df_t.loc[df_t.smpl.str.contains("HEPAR|hep"),'lbl'] = 1
  df = df_t
elif args.experiment == 'comb3': ##HCV_hepar vs TCCSUP_ctrl
  df_t1 = df.loc[(df.smpl.str.contains("HCV")) & (df.smpl.str.contains("HEPAR|hep"))]
  df_t2 = df.loc[(df.smpl.str.contains("TCCSUP")) & ~(df.smpl.str.contains("HEPAR|hep"))]
  df_t = pd.concat((df_t1,df_t2))
  df_t['lbl'] = 0
  df_t.loc[df_t.smpl.str.contains("HEPAR|hep"),'lbl'] = 1 ## HCV_ctrl is 1, TCCSUP_hepar is 0
  df = df_t
elif args.experiment == 'comb4': ##HCV_ctrl vs TCCSUP_hepar
  df_t1 = df.loc[(df.smpl.str.contains("HCV")) & ~(df.smpl.str.contains("HEPAR|hep"))]
  df_t2 = df.loc[(df.smpl.str.contains("TCCSUP")) & (df.smpl.str.contains("HEPAR|hep"))]
  df_t = pd.concat((df_t1,df_t2))
  df_t['lbl'] = 1
  df_t.loc[df_t.smpl.str.contains("HEPAR|hep"),'lbl'] = 0 ## HCV_ctrl is 1, TCCSUP_hepar is 0
  df = df_t
  df = df.drop(index=np.random.choice(df.loc[df.lbl==0].index.to_numpy(),20,replace=False)) ## we have too many HEPAR TCCSUP samples here

print("Label 0 samples",df.loc[df.lbl==0].shape)
print("Label 1 samples",df.loc[df.lbl==1].shape)

X = df.iloc[:,:-2]
feat_names = X.columns.to_list()
X = X.to_numpy()
y = df.iloc[:,-1].to_numpy()


#####Block for Gini importnces#########
def gini_importance_multirun(X, y, test_size,var_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    ########## Choice of the feature selection method matters !
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    #forest = RandomForestClassifier(max_depth=5, n_estimators=100)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    fx_imp = importances#pd.Series(importances, index=var_names)
    return fx_imp

def gini_importance_parallel(X, y, test_size, var_names, n_cpu_cores, n_runs):
    importances = joblib.Parallel(n_jobs=n_cpu_cores)(
        joblib.delayed(gini_importance_multirun)(
            X, y, test_size,var_names) for n in range(n_runs))
    return pd.DataFrame(importances,columns=feat_names)#importances

def plot_Gini(imp,name):
  mean_imp = np.mean(imp.values, axis=0)
  std_imp = np.std(imp.values, axis=0) # error bars
  indices = np.argsort(mean_imp)[::-1] # 
  feat_names = imp.columns.tolist()

  label_ranking = np.array(feat_names)[indices]
  print("Top 10",label_ranking[:10])


  # plt.figure(facecolor='white', figsize=(8,10))
  # num_vars = 20#len(feat_names)
  # plt.barh(range(num_vars), mean_imp[indices][:num_vars], xerr=std_imp[indices][:num_vars],
  #          align='center', tick_label=np.array(feat_names)[indices][:num_vars])
  # plt.ylim([-1, num_vars])
  # #plt.xlabel('Importance coefficient', fontsize=12)
  # #plt.gca().set_yticklabels(tick_labels, fontsize=10)
  # plt.gca().invert_yaxis()
  # plt.tick_params(axis='both', which='both', left='off', right='off', top='off')

  #plt.savefig('./%s.png'%(name))

  return indices, label_ranking
######################################
#########Classification ##########
# def get_acc(rank, X_sorted, y, test_size):
#   classifier = RandomForestClassifier(max_depth=5, n_estimators=100)
#   #areas = []
#   accuracies = []
#   for _ in range(500):
#       X_train, X_test, y_train, y_test = train_test_split(X_sorted[:,:rank], y, test_size=test_size, stratify=y)
      
#       classifier.fit(X_train, y_train)
#       probs = classifier.predict_proba(X_test)[:, 1]
#       #auc = roc_auc_score(y_test, probs)

#       #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#       #areas.append(auc)
#       acc = classifier.score(X_test, y_test)
#       accuracies.append(acc)
#   return np.mean(np.array(accuracies)), np.std(np.array(accuracies))

def get_acc_multirun(rank, X_sorted, y, test_size):
  X_train, X_test, y_train, y_test = train_test_split(X_sorted[:,:rank], y, test_size=test_size, stratify=y)
  classifier = RandomForestClassifier(max_depth=5, n_estimators=100)

  classifier.fit(X_train, y_train)
  probs = classifier.predict_proba(X_test)[:, 1]
  acc = classifier.score(X_test, y_test)
  auc = roc_auc_score(y_test, probs)
  return acc, auc

def get_acc_parallel(rank, X_sorted, y, test_size,n_cpu_cores, n_runs, label_ranking):
  print("The features used:", label_ranking[:rank])
  metrs = joblib.Parallel(n_jobs=n_cpu_cores)(
      joblib.delayed(get_acc_multirun)(
          rank, X_sorted[:,:rank], y, test_size) for n in range(n_runs))
  accs = np.array(metrs)[:,0]
  aucs = np.array(metrs)[:,1]
  #return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)
  return accs, aucs

######################################  

if args.manual_features == 'manual':
  X_sorted = X
  label_ranking = df.columns.to_list()
else:
  test_sizeGini=0.3
  var_imp_series = gini_importance_parallel(X, y, test_sizeGini, feat_names, n_cpu_cores, 500)
  indices, label_ranking = plot_Gini(var_imp_series,args.name)
  print("Done with data reduction")

  X_sorted = X[:,indices] ## All the features, sorted by Gini
  #X_red = X_sorted[:,:args.rank]



if args.type == 'one_acc':
  print("This is simple accuracy experiment.")
  accs, aucs = get_acc_parallel(args.rank, X_sorted, y, args.test_size, n_cpu_cores, 500, label_ranking) ##parallel_version
  print("Results for experiment: ", args.experiment, ". Channel: ", args.input_channel)
  print("ACC: ", np.mean(accs), " +/- ", np.std(accs))
  print("AUC: ", np.mean(aucs), " +/- ", np.std(aucs))
  if args.verbose == "high":
    #pd.DataFrame({'acc':accs,'auc':aucs}).to_csv('./perf_metrics_full_%s.csv' %(args.name))
    pd.DataFrame({'auc':aucs}).to_csv('./perf_metrics_full_%s.csv' %(args.name))


if args.type == 'acc_vs_nfeat':
  print("This is accuracy vs number of features experiment.")
  #feat_n = np.arange(1,11,1)
  feat_n = np.arange(1,df.shape[1]-1,1)
  #feat_n = np.array([1,2])
  print("The numbers of features we use are:", feat_n)
  accs = []
  stddevs = []
  aucs = []
  aucs_sd = []
  AUC = [] ## full AUC array which is not averaged over train/test splits
  for n in feat_n:
    #mean, stddev, auc_m, auc_sd, 
    accs_arr, aucs_arr = get_acc_parallel(n, X_sorted, y, args.test_size, n_cpu_cores, 500, label_ranking) ##parallel_version
    print(np.mean(accs_arr), np.mean(aucs_arr))
    #mean, stddev = get_acc(n, X_sorted, y, args.test_size)
    accs.append(np.mean(accs_arr))
    stddevs.append(np.std(accs_arr))
    aucs.append(np.mean(aucs_arr))
    aucs_sd.append(np.std(aucs_arr))
    if args.verbose == 'high':
      AUC.append(aucs_arr)

  if args.verbose == 'high':
    AUC = np.array(AUC)
    pd.DataFrame(AUC.transpose(),columns=feat_n).to_csv('./perf_metrics_full_%s.csv' %(args.name))
  accs = np.array(accs)
  stddevs = np.array(stddevs)
  aucs = np.array(aucs)
  aucs_sd = np.array(aucs_sd)
  ### Pretty plot of the results
  plt.style.use('ggplot') #Change/Remove This If you Want

  fig, ax = plt.subplots(figsize=(8, 4))
  #ax.plot(feat_n, accs, alpha=0.5, color='red', label='acc_vs_nfeat', linewidth = 1.0)
  #ax.plot(feat_n, accs, 'x')

  ax.plot(feat_n, aucs, alpha=0.5, color='blue', label='acc_vs_nfeat', linewidth = 1.0)
  ax.plot(feat_n, aucs, 'x')

  #ax.fill_between(feat_n, accs - stddevs, accs + stddevs, color='red', alpha=0.2)
  ax.fill_between(feat_n, aucs - aucs_sd, aucs + aucs_sd, color='blue', alpha=0.2)
  ax.set_ylim([0.5,1])
  #ax.set_ylabel("Accuracy/AUC")
  ax.set_ylabel("AUC")
  ax.set_xlabel("N_features")
  plt.savefig('./accauc_vs_nfeat_%s.png'%(args.name))

  ### Save the accuracies in a file
  dict_df = {'acc': np.round(accs,2), 'acc_stddev': np.round(stddevs,2),
  'auc': np.round(aucs,2), 'auc_stddev': np.round(aucs_sd,2)}
  ds_accs = pd.DataFrame(dict_df)
  ds_accs.to_excel('./accauc_vs_nfeat_%s.xlsx'%(args.name))
  print("Experiment is done.")








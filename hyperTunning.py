scores=[]
hyperparams=[]
for i in range(3):
  for e in range(3):
    for z in range(3):

      # "Technical" parameters of the model:
      params = {'objective': 'Logloss',
        'learning_rate': grid['learning_rate'][i], # learning rate, lower -> slower but better prediction
        'depth': grid['max_depth'][e], # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
        'min_data_in_leaf': 10,
        'l2_leaf_reg': grid['l2_leaf_reg'][z], # L2 regularization (between 3 and 20, higher -> less overfitting)
        'rsm': 0.90, # % of features to consider in each split (lower -> faster and reduces overfitting)
        'subsample': 0.95, # Sample rate for bagging
        'random_seed': RS}
      print(params)
      print('\nCatboost CV...')
      print('########################################################')
      
      for nrounds in iter:

        print('\nn rounds: ',nrounds)

        # Define the model
        model_catboost_cv=CatBoostClassifier()
        model_catboost_cv.set_params(**params)
        model_catboost_cv.set_params(n_estimators=nrounds)
        model_catboost_cv.set_params(verbose=False)

        Pred_train, Pred_test, s = Model_cv(model_catboost_cv,n_folds,X_train,X_test,Y_train,RS,makepred=True,CatPos=Pos)

        # Look if we are in the first test:
        if len(scores)==0:
          max_score=float('-inf')
        else:
          max_score=max(scores)

        # If the score improves, we keep this one:
        if s>=max_score:
          print('BEST')
          Catboost_train=Pred_train.copy()
          Catboost_test=Pred_test.copy()

        # Append score
        scores.append(s)

      # The best cross-validated score has been found in:
      print('\n###########################################')
      print('Catboost optimal rounds: ',iter[scores.index(max(scores))])
      print('Catboost optimal GINI: ',round((max(scores)*2-1)*100,4),'%')
      print('Catboost optimal AUC: ',round(max(scores)*100,4),'%')
      print('###########################################')
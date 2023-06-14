from teradataml import copy_to_sql, DataFrame, get_context, get_connection
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext
)

import joblib
import pandas as pd


def score(context: ModelContext, **kwargs):
    aoa_create_context()
    query = '''
create multiset table DF_predict_test as (
    SELECT * FROM TD_XGBoostPredict(
        ON pima_patient_features_test as inputtable partition by ANY
        ON xgboost_classification_model as modeltable dimension order by task_index, tree_num, iter, class_num, tree_order
        USING
         IdColumn('PatientId')
         ModelType('Classification')
         OutputProb('t')
         Responses('1','0')
         Accumulate('"hasdiabetes"')
        ) as dt
) with data;
'''
    
    try:
        #score_df = DataFrame.from_query(query)
        get_context().execute(query)
    except:
        get_context().execute('DROP TABLE DF_predict_test;')
        #score_df = DataFrame.from_query(query)
        get_context().execute(query)

    print(pd.read_sql('SELECT * FROM DF_predict_test', get_connection()))
    print('Scoring Complete')

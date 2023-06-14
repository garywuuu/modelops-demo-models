from teradataml import (
    DataFrame,
    GLM,
    ScaleFit,
    ScaleTransform,
    get_context,
    get_connection
)

import pandas as pd
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np

def train(context: ModelContext, **kwargs):
    aoa_create_context()
    query = '''CREATE multiset table DF_table as 
                ( SELECT * FROM TD_XGBoost 
                ( ON pima_patient_features_train partition by ANY 
                USING 
                    ResponseColumn('"HasDiabetes"') 
                    InputColumns('[1:8]') 
                    Modeltype('classification') 
                    RegularizationLambda(1000)  
                    ShrinkageFactor(0.1) 
                    IterNum(10) 
                    MinNodeSize(1) 
                    MaxDepth(12) 
                ) as dt) with data;'''

    try:
        model_df = DataFrame.from_query(f'''SELECT numtimesprg,PlGlcConc,bloodp,skinthick,twoHourSerIns,BMI,DiPedFunc,Age,"HasDiabetes" FROM pima_patient_features_train;''')
        get_context().execute(query)
    except:
        get_context().execute('DROP TABLE DF_table;')
        model_df = DataFrame.from_query(f'''SELECT numtimesprg,PlGlcConc,bloodp,skinthick,twoHourSerIns,BMI,DiPedFunc,Age,"HasDiabetes" FROM pima_patient_features_train;''')
        get_context().execute(query)
        
    print(pd.read_sql('select * from DF_table', get_connection()))
    print('Training Complete')

    feature_names = ["NumTimesPrg", "PlGlcConc", "BloodP", "SkinThick", "TwoHourSerIns", "BMI", "DiPedFunc", "Age"]
    target_name = ["hasdiabetes"]
    
    record_training_stats(model_df,
                           features=feature_names,
                           categorical=target_name,
                           targets=target_name,
                           context=context)

from sklearn import metrics
import matplotlib.pyplot as plt
from teradataml import DataFrame, copy_to_sql, get_context, get_connection, ScaleTransform
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd
def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()

    query = '''

    SELECT * from TD_CLASSIFICATIONEVALUATOR(
        ON DF_predict_test AS InputTable
        OUT VOLATILE TABLE OutputTable(additional_metrics_diabetes_test)
        USING
            Labels(1,0)
            ObservationColumn('hasdiabetes')
            PredictionColumn ('prediction')
            ) as dt1; 
        
    '''

    try:
        #eval_df = DataFrame.from_query(query)
        get_context().execute(query)
    except:
        get_context().execute('DROP TABLE additional_metrics_diabetes_test;')
        #eval_df = DataFrame.from_query(query)
        get_context().execute(query)

    eval_df = pd.read_sql('SELECT * FROM additional_metrics_diabetes_test', get_connection())
    eval_df
    print('Evaluation Complete')

    test_df = pd.read_sql("SELECT * FROM pima_patient_features_test", get_connection())
    #     print(test_df)
    out_df = pd.read_sql("SELECT * FROM DF_predict_test", get_connection())
    #     print(out_df)
    fpr, tpr, threshold = metrics.roc_curve(test_df["HasDiabetes"], out_df["Prob_1"])
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Evaluation Diabetes')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('tp_vs_fp.png')
    
    #save plot here

#     model = DataFrame(f"model_${context.model_version}")

#     feature_names = context.dataset_info.feature_names
#     target_name = context.dataset_info.target_names[0]
#     entity_key = context.dataset_info.entity_key

#      test_df = DataFrame.from_query(context.dataset_info.sql)

#     # Scaling the test set
#     print ("Loading scaler...")
#     scaler = DataFrame(f"scaler_${context.model_version}")

#     scaled_test = ScaleTransform(
#         data=test_df,
#         object=scaler,
#         accumulate = [target_name,entity_key]
#     )
    
#     print("Scoring")
#     predictions = TDXGBOOSTPredict(
#         object=model,
#         newdata=scaled_test.result,
#         accumulate=target_name,
#         id_column=entity_key,
#         output_prob=True,
#         output_responses=['0','1']
#     )

#     predicted_data = ConvertTo(
#         data = predictions.result,
#         target_columns = [target_name,'prediction'],
#         target_datatype = ["INTEGER"]
#     )

#     ClassificationEvaluator_obj = ClassificationEvaluator(
#         data=predicted_data.result,
#         observation_column=target_name,
#         prediction_column='prediction',
#         num_labels=2
#     )

#     metrics_pd = ClassificationEvaluator_obj.output_data.to_pandas()

#     evaluation = {
#         'Accuracy': '{:.2f}'.format(metrics_pd.MetricValue[0]),
#         'Micro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[1]),
#         'Micro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[2]),
#         'Micro-F1': '{:.2f}'.format(metrics_pd.MetricValue[3]),
#         'Macro-Precision': '{:.2f}'.format(metrics_pd.MetricValue[4]),
#         'Macro-Recall': '{:.2f}'.format(metrics_pd.MetricValue[5]),
#         'Macro-F1': '{:.2f}'.format(metrics_pd.MetricValue[6]),
#         'Weighted-Precision': '{:.2f}'.format(metrics_pd.MetricValue[7]),
#         'Weighted-Recall': '{:.2f}'.format(metrics_pd.MetricValue[8]),
#         'Weighted-F1': '{:.2f}'.format(metrics_pd.MetricValue[9]),
#     }

#     with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
#         json.dump(evaluation, f)
        
#     cm = confusion_matrix(predicted_data.result.to_pandas()['HasDiabetes'], predicted_data.result.to_pandas()['prediction'])
#     plot_confusion_matrix(cm, f"{context.artifact_output_path}/confusion_matrix")

#     roc_out = ROC(
#         data=predictions.result,
#         probability_column='prob_1',
#         observation_column=target_name,
#         positive_class='1',
#         num_thresholds=1000
#     )
#     plot_roc_curve(roc_out, f"{context.artifact_output_path}/roc_curve")

#     # Calculate feature importance and generate plot
#     model_pdf = model.to_pandas()[['predictor','estimate']]
#     predictor_dict = {}
    
#     for index, row in model_pdf.iterrows():
#         if row['predictor'] in feature_names:
#             value = row['estimate']
#             predictor_dict[row['predictor']] = value
    
#     feature_importance = dict(sorted(predictor_dict.items(), key=lambda x: x[1], reverse=True))
#     keys, values = zip(*feature_importance.items())
#     norm_values = (values-np.min(values))/(np.max(values)-np.min(values))
#     feature_importance = {keys[i]: float(norm_values[i]*1000) for i in range(len(keys))}
#     plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")

#     predictions_table = "predictions_tmp"
#     copy_to_sql(df=predicted_data.result, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

#     # calculate stats if training stats exist
#     if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
#         record_evaluation_stats(
#             features_df=test_df,
#             predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
#             feature_importance=feature_importance,
#             context=context
#         )

# def plot_feature_importance(fi, img_filename):
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     feat_importances = pd.Series(fi)
#     feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
#     fig = plt.gcf()
#     fig.savefig(img_filename, dpi=500)
#     plt.clf()
    
    
# def plot_confusion_matrix(cf, img_filename):
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(figsize=(7.5, 7.5))
#     ax.matshow(cf, cmap=plt.cm.Blues, alpha=0.3)
#     for i in range(cf.shape[0]):
#         for j in range(cf.shape[1]):
#             ax.text(x=j, y=i,s=cf[i, j], va='center', ha='center', size='xx-large')
#     ax.set_xlabel('Predicted labels');
#     ax.set_ylabel('True labels'); 
#     ax.set_title('Confusion Matrix');
#     fig = plt.gcf()
#     fig.savefig(img_filename, dpi=500)
#     plt.clf()

    
# def plot_roc_curve(roc_out, img_filename):
#     import matplotlib.pyplot as plt
#     auc = roc_out.result.to_pandas().reset_index()['AUC'][0]
#     roc_results = roc_out.output_data.to_pandas()
#     plt.plot(roc_results['fpr'], roc_results['tpr'], color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % 0.27)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     fig = plt.gcf()
#     fig.savefig(img_filename, dpi=500)
#     plt.clf()

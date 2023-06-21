##############################################################################
# Import necessary modules
# #############################################################################


from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta
import pandas as pd
import importlib.util

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = module_from_file('utils','/home/airflow/dags/Lead_Scoring_Data_Pipeline/utils.py')
data_validation_checks = module_from_file('data_validation_checks','/home/airflow/dags/Lead_Scoring_Data_Pipeline/data_validation_checks.py')

###############################################################################
# Define default arguments and DAG
###############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


ML_data_cleaning_dag = DAG(
                dag_id = 'Lead_Scoring_Data_Engineering_Pipeline',
                default_args = default_args,
                description = 'DAG to run data pipeline for lead scoring',
                schedule_interval = '@daily',
                catchup = False
)

###############################################################################
# Create a task for build_dbs() function with task_id 'building_db'
###############################################################################
building_db = PythonOperator(
            task_id = 'building_db',
            python_callable = utils.build_dbs,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for raw_data_schema_check() function with task_id 'checking_raw_data_schema'
###############################################################################
checking_raw_data_schema = PythonOperator(
            task_id = 'checking_raw_data_schema',
            python_callable = data_validation_checks.raw_data_schema_check,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for load_data_into_db() function with task_id 'loading_data'
##############################################################################
loading_data = PythonOperator(
            task_id = 'loading_data',
            python_callable = utils.load_data_into_db,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for map_city_tier() function with task_id 'mapping_city_tier'
###############################################################################
mapping_city_tier = PythonOperator(
            task_id = 'mapping_city_tier',
            python_callable = utils.map_city_tier,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for map_categorical_vars() function with task_id 'mapping_categorical_vars'
###############################################################################
mapping_categorical_vars = PythonOperator(
            task_id = 'mapping_categorical_vars',
            python_callable = utils.map_categorical_vars,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for interactions_mapping() function with task_id 'mapping_interactions'
###############################################################################
mapping_interations = PythonOperator(
            task_id = 'mapping_interactions',
            python_callable = utils.interactions_mapping,
            dag = ML_data_cleaning_dag)
###############################################################################
# Create a task for model_input_schema_check() function with task_id 'checking_model_inputs_schema'
###############################################################################
checking_model_inputs_schema = PythonOperator(
            task_id = 'checking_model_inputs_schema',
            python_callable = data_validation_checks.model_input_schema_check,
            dag = ML_data_cleaning_dag)
###############################################################################
# Define the relation between the tasks
###############################################################################

building_db >> checking_raw_data_schema >> loading_data >> mapping_city_tier >> mapping_categorical_vars >> mapping_interations >> checking_model_inputs_schema



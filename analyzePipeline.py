from enum import auto
import boto3
import time
import json
from cv2 import dft
import pandas as pd
import numpy as np
from sqlalchemy import true
from trp import Document
from boto3.dynamodb.conditions import Key

import re
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

from decimal import Decimal

s3 = boto3.client("s3")
textract = boto3.client("textract")
dynamodb = boto3.resource('dynamodb', region_name="us-east-1")
rekognition  = boto3.client('rekognition')


TABLE_NAME = "VaccineInputUS-a7zevy56jbg5vhz7otjz2zgu6i-staging"
# Creating the DynamoDB Client
dynamoDb_client = boto3.client('dynamodb')
# Creating the DynamoDB Table Resource
input_table = dynamodb.Table(TABLE_NAME)
output_table = dynamodb.Table("VaccineOutputUS-a7zevy56jbg5vhz7otjz2zgu6i-staging")
s3BucketVaccineCards = "unitedairlineshealth-storage232654-staging"
# s3BucketVaccineCards = "demovaccinecards2022"               # REPLACE WITH S3 BUCKET NAME
# vaccineCardFile = "Covid_Vaccine_Card_old.jpg"             # REPLACE FOR CARD IN BUCKET




# Scans DynamoDB table to get user information
# https://alexwlchan.net/2020/05/getting-every-item-from-a-dynamodb-table-with-python/
def scan_table(dynamo_client, *, TableName, **kwargs):
    """
    Generates all the items in a DynamoDB table.

    :param dynamo_client: A boto3 client for DynamoDB.
    :param TableName: The name of the table to scan.

    Other keyword arguments will be passed directly to the Scan operation.
    See https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/dynamodb.html#DynamoDB.Client.scan

    """
    paginator = dynamo_client.get_paginator("scan")

    for page in paginator.paginate(TableName=TableName, **kwargs):
        yield from page["Items"]

# Calls the textract service
def start_analyze(s3BucketVaccineCards, vaccineCardFile, feature_type):
    doc_spec = {"S3Object": {"Bucket": s3BucketVaccineCards, "Name": vaccineCardFile}}

    response = textract.start_document_analysis(
        DocumentLocation=doc_spec, FeatureTypes=[feature_type]
    )
    return response["JobId"]

def CheckAnalyzeJobComplete(jobId):
    time.sleep(5)
    response = textract.get_document_analysis(JobId=jobId)
    status = response["JobStatus"]
    print("Job status: {}".format(status))
    while(status == "IN_PROGRESS"):
        time.sleep(5)
        response = textract.get_document_analysis(JobId=jobId)
        status = response["JobStatus"]
        print("Job status: {}".format(status))
    return status

# Gets Textract results
def get_textract_results(job_id):
    response = textract.get_document_analysis(JobId=job_id)
    pages = [response]

    while "NextToken" in response:
        time.sleep(0.25)

        response = textract.get_document_analysis(
            JobId=job_id, NextToken=response["NextToken"]
        )

        pages.append(response)

    return pages

# https://betterprogramming.pub/extract-data-from-pdf-files-using-aws-textract-with-python-12ba62fde1b0
############################
     # TABLE ANALYZE #
############################

def get_textract_tables(job_id):
    response = textract.get_document_analysis(JobId=job_id)
    doc = Document(response)
    for page in doc.pages:
        for table in page.tables:
            data = [[cell.text for cell in row.cells] for row in table.rows]
            df = pd.DataFrame(data)

    return df

# Primary Table Extraction Function
def runTableAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile):
    textractJobId = start_analyze(s3BucketVaccineCards, vaccineCardFile, "TABLES")
    print("Started job with id: {}".format(textractJobId))
    
    if(CheckAnalyzeJobComplete(textractJobId)):
        df = get_textract_tables(textractJobId)

    return df

############################
      # FORM ANALYZE #
############################

# https://www.crosstab.io/articles/amazon-textract-review
def filter_key_blocks(blocks: dict) -> list:
    """Identify blocks that are keys in extracted key-value pairs."""
    return [
        k
        for k, v in blocks.items()
        if v["BlockType"] == "KEY_VALUE_SET" and "KEY" in v["EntityTypes"]
    ]

def identify_block_children(block: dict) -> list:
    """Extract the blocks IDs of the given block's children.
    Presumably, order matters here, and the order needs to be maintained through text
    concatenation to get the full key text.
    """

    child_ids = []

    if "Relationships" in block.keys():
        child_ids = [
            ix
            for link in block["Relationships"]
            if link["Type"] == "CHILD"
            for ix in link["Ids"]
        ]

    return child_ids

def concat_block_texts(blocks: list) -> str:
    """Combine child block texts to get the text for an abstract block."""
    return " ".join([b["Text"] for b in blocks])

def identify_value_block(block: dict) -> str:
    """Given a key block, find the ID of the corresponding value block."""
    return [x for x in block["Relationships"] if x["Type"] == "VALUE"][0]["Ids"][0]

def get_form_dataframe(blocks):
    results = []
    key_ids = filter_key_blocks(blocks)
    for k in key_ids:
        child_ids = identify_block_children(blocks[k])
        child_blocks = [blocks[c] for c in child_ids]
        key_text = concat_block_texts(child_blocks)

        v = identify_value_block(blocks[k])
        child_ids = identify_block_children(blocks[v])
        child_blocks = [blocks[c] for c in child_ids]
        value_text = concat_block_texts(child_blocks)

        result = {
            #"key_id": k,
            "key_text": key_text,
            "key_confidence": blocks[k]["Confidence"],
            #"value_id": v,
            "value_text": value_text,
            # "value_confidence": blocks[v]["Confidence"],
        }

        results.append(result)

    df = pd.DataFrame(results)
    if len(df)== 0:
        return df
    df = df[df.value_text != '']
    return df

# Primary Form Extraction 
def runFormAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile):

    textractJobId = start_analyze(s3BucketVaccineCards, vaccineCardFile, "FORMS")
    print("Started job with id: {}".format(textractJobId))
    
    if(CheckAnalyzeJobComplete(textractJobId)):
        pages = get_textract_results(textractJobId)

    blocks = {block["Id"]: block for page in pages for block in page["Blocks"]}

    return get_form_dataframe(blocks)
    

############################
      # AUTOCORRECT #
############################

def autocorrect(input, correct_words, view_tags=False):

    dis = 1000
    correct = input
    key = input
    for word in correct_words.keys():
        # https://python.gotrained.com/nltk-edit-distance-jaccard-distance/
        ed = nltk.edit_distance(input.lower(), word)
        if ed < dis and ed < len(input.strip()) and ed < len(word):
            dis = ed
            correct = word
            key = correct_words[correct]
            continue

        match = re.search(r'(\d+/\d+/\d+)', input)
        if bool(match):
            key = match.group(1)
            if view_tags:
                print('date: ', match.group(1))
            break

        #if ("/" in input or "-" in input):
        #    if (input[0].isdigit()):
        #        print('date: ', input)
    if view_tags:
        print(key, ': ', correct, dis)
    return key

# Autocorrects table
def correct_all_table(df):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print("table_df:")
    # print(df)
    correct_words = {'pfizer':'vaccine$ pfizer', 'pfizer xxxxxx':'vaccine$ pfizer', 'pfizer-biontech': 'vaccine$ pfizer', 
        'moderna':'vaccine$ moderna', '1st dose':'dose1', '1st dose covid-19': 'dose1', '2nd Dose': 'dose2', 
        '2nd dose covid-19': 'dose2', 'walgreens': 'walgreens', 'date': 'Date Header',
        'product name/manufacturer lot number': 'Manufacturer Header','lot number': 'Manufacturer Header', 'vaccine': 'Vaccine Header',
        'healthcare professional or clinic site': 'Site Header','or clinic site': 'Site Header', 'other': 'none', 'mm dd yy': 'none'}

    #print(df)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            # https://www.stackvidhya.com/get-value-of-cell-from-a-pandas-dataframe/
            if type(df.iat[row,col]) == str:
                if "pfizer" in df.iat[row,col].lower():
                    df.iat[row,col] = "vaccine$ pfizer"
                if "moderna" in df.iat[row,col].lower():
                    df.iat[row,col] = "vaccine$ moderna"
                df.iat[row,col] = autocorrect(df.iat[row,col], correct_words, False)
    return df

# Deletes unnecessary chars
def delete_dates(inputString):
    to_return = ""
    unwanted_digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "/"]
    for char in inputString:
        if (not (char in unwanted_digits)):
            to_return += char
    if (to_return == ""):
        return "N/A"
    return to_return


def format_as_date(updated_dose1_date, updated_dose2_date, updated_dob_date):
    # List of characters to remove (including space)
    characters_to_remove = "~-_+=\{\}'\".,?\:;<>[]ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()/ "
    characters_to_remove = list(characters_to_remove)
    # Delete characters (replace each char with "")
    for character in characters_to_remove:
        updated_dose1_date = updated_dose1_date.replace(character, "")
        updated_dose2_date = updated_dose2_date.replace(character, "")
        updated_dob_date = updated_dob_date.replace(character, "")
    # Reverse the dates
    updated_dose1_date=updated_dose1_date[len(updated_dose1_date)::-1]
    updated_dose2_date=updated_dose2_date[len(updated_dose2_date)::-1]
    updated_dob_date=updated_dob_date[len(updated_dob_date)::-1]
    # Add a slash after every two characters
    updated_dose1_date = '/'.join(updated_dose1_date[i:i + 2] for i in range(0, len(updated_dose1_date), 2))
    updated_dose2_date = '/'.join(updated_dose2_date[i:i + 2] for i in range(0, len(updated_dose2_date), 2))
    updated_dob_date = '/'.join(updated_dob_date[i:i + 2] for i in range(0, len(updated_dob_date), 2))
    # Reverse the dates again
    updated_dose1_date = updated_dose1_date[len(updated_dose1_date)::-1]
    updated_dose2_date = updated_dose2_date[len(updated_dose2_date)::-1]
    updated_dob_date = updated_dob_date[len(updated_dob_date)::-1]
    # Account for mm/dd/yyyy format
    if len(updated_dose1_date) >= 10:
        extra_slash_index = updated_dose1_date.rfind("/")
        updated_dose1_date = updated_dose1_date[:extra_slash_index] + updated_dose1_date[extra_slash_index+1:]
    if len(updated_dose2_date) >= 10:
        extra_slash_index = updated_dose2_date.rfind("/")
        updated_dose2_date = updated_dose2_date[:extra_slash_index] + updated_dose2_date[extra_slash_index+1:]
    if len(updated_dob_date) >= 10:
        extra_slash_index = updated_dob_date.rfind("/")
        updated_dob_date = updated_dob_date[:extra_slash_index] + updated_dob_date[extra_slash_index+1:]
    # Account for m/d/yy format
    if len(updated_dose1_date) <= 5 and len(updated_dose1_date) > 0:
        updated_dose1_date = updated_dose1_date[:1] + "/" + updated_dose1_date[1:]
    if len(updated_dose2_date) <= 5 and len(updated_dose2_date) > 0:
        updated_dose2_date = updated_dose2_date[:1] + "/" + updated_dose2_date[1:]
    if len(updated_dob_date) <= 5 and len(updated_dob_date) > 0:
        updated_dob_date = updated_dob_date[:1] + "/" + updated_dob_date[1:]
    # Add all the updated dates to an array and return
    updated_dates = []
    updated_dates.extend((updated_dose1_date, updated_dose2_date, updated_dob_date))
    return updated_dates

def validateCard(fileName):
    response = rekognition.detect_labels(
        Image={
            'Bytes' : b'bytes',
            'S3Object': {
                'Bucket': s3BucketVaccineCards,
                'Name': fileName,
            }
        },
        MaxLabels=2,
        MinConfidence=80
    )

################### FINAL DF ############################
# Creates final_df, merging form and table extraction along with autocorrection and flagging 
def create_final_df(vaccine_card):
    # Gets form_df
    form_df = runFormAnalyzeTextract(s3BucketVaccineCards, vaccine_card)
    
    # form_df is empty --> card is not valid
    if(len(form_df)==0):
        print("Image is not valid!")
        return form_df

    form_df.set_index('key_text', inplace=True)
    fields = ['First Name','Last Name','Date of birth']
    f_df = pd.DataFrame()
    conf = 0
    for i in range(len(fields)):
        if fields[i] in form_df.index:
            f_df[fields[i]] =(form_df.iloc[i].loc['value_text']).split(',') 

            conf+= int((form_df.iloc[i].loc['key_confidence']))
        else:   # Replaces any missing values with N/A
            f_df[fields[i]] = 'N/A'

    # Gets table_df
    table_df = runTableAnalyzeTextract(s3BucketVaccineCards, vaccine_card)
    print("table_df",table_df)
    corrected_df = correct_all_table(table_df)

    # Makes sure that first row has the headers we are looking for 
    if corrected_df[0][0]!='Vaccine Header':
        corrected_df = corrected_df[1:]
    # print(corrected_df)
   
    new_header = corrected_df.iloc[0] #grab the first row for the header
    corrected_df = corrected_df[1:] #take the data less the header row
    corrected_df.columns = new_header #set the header row as the df header
    
    #created separate dfs for the doses because they need to have different column names in the final df
    dose1_df = pd.DataFrame()
    dose2_df = pd.DataFrame()
    if 'Vaccine Header' in corrected_df.columns:
        dose1_df = corrected_df.loc[corrected_df['Vaccine Header'] == 'dose1']
        dose1_df = dose1_df.drop("Vaccine Header", axis=1)
        dose2_df = corrected_df.loc[corrected_df['Vaccine Header'] == 'dose2']
        dose2_df = dose2_df.drop("Vaccine Header", axis=1)
    else:
        dose1_df = dose1_df.iloc[1:]
        dose1_df = dose1_df.iloc[2:]

    dose1_df = dose1_df.rename(columns={'Manufacturer Header': 'dose1_manufacturer', 'Date Header': 'dose1_date', 'Site Header': 'dose1_location' })
    dose2_df = dose2_df.rename(columns={'Manufacturer Header': 'dose2_manufacturer', 'Date Header': 'dose2_date', 'Site Header': 'dose2_location' })
 
    frames = [dose1_df, dose2_df]
    t_df = pd.concat(frames)
    t_df = t_df.fillna(method='bfill')
    t_df = t_df[:-1]

    #final_df contains form AND table dfs
    final_frames = [f_df, t_df]
    final_df = pd.concat(final_frames)
    # print(final_df)
    final_df = final_df.fillna(method='bfill')
    final_df = final_df[:-1]
    
    # Removes any unnecessary columns
    columns_required = ["First Name","Last Name", "Date of birth","dose1_date","dose1_manufacturer","dose1_location","dose2_date","dose2_manufacturer","dose2_location","Flag"]
    for col in final_df.columns:
        if col not in columns_required:
            final_df.drop(col,inplace = True, axis = 1)

    # Make sure dose1_manufacturer and dose2_manufacturer don't contain dates (numbers or slashes)
    final_df['dose1_manufacturer'] = delete_dates(final_df['dose1_manufacturer'])
    final_df['dose2_manufacturer'] = delete_dates(final_df['dose2_manufacturer'])
    
    # Update dates array using format_as_date function 
    # + Removes all special characters, spaces
    # + Re-adds slashes starting from back of the date to achieve desired formatting
    if (final_df['dose1_date'][0] is not None and final_df['dose2_date'][0] is not None):
        updated_dates = format_as_date(final_df['dose1_date'][0], final_df['dose2_date'][0], final_df['Date of birth'][0])

    # Replace the dates with their updated versions in final_df
    final_df['dose1_date'][0] = updated_dates[0]
    final_df['dose2_date'][0] = updated_dates[1]
    final_df['Date of birth'][0] = updated_dates[2]

    final_df = final_df.replace(np.nan, "N/A")
    final_df = final_df.replace('', "N/A")


     # Flags vaccine card (True) if Vaccine card extraction contains any N/A values
    flagged_cols = []
    for i in range(len(final_df.columns)):
        if (final_df[final_df.columns[i]].str.contains("N/A").any()):
            flagged_cols.append(i)
    final_df["Flag"] = str(flagged_cols)[1:-1]

    ##### Computes confidence level as average of confidence levels of fields in form df
    final_df["Confidence Level"] = conf/len(fields)


    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(final_df)
    return final_df

# def checkImageInBucket(imageName, s3bucket):
#     for object in s3bucket.objects.filter(Prefix="public/VaccineInputUS/"):
#         if object.key == imageName:
#             return True
#     return False

# Adds a users output to table
def addOutputToTable(output,user):
    output_table.put_item(
        Item = {
            'id': user['id']['S'],
            'User_Email': user['User_Email']['S'],
            'Form_Name': user['Form_Name']['S'],
            'First_Name': output['First Name'][0],
            'Last_Name': output['Last Name'][0],
            'Date_of_Birth': output['Date of birth'][0],
            'Dose1_Manufacturer': output['dose1_manufacturer'][0],
            'Dose1_Date': output['dose1_date'][0],
            'Dose1_Location': output['dose1_location'][0],
            'Dose2_Manufacturer': output['dose2_manufacturer'][0],
            'Dose2_Date': output['dose2_date'][0],
            'Dose2_Location': output['dose2_location'][0],
            'Flag': output['Flag'][0],
            'Confidence_Level' : Decimal(str(output['Confidence Level'][0]))
        }
    )


def run():
    print("Input Table")
    for item in scan_table(dynamo_client = dynamoDb_client, TableName = TABLE_NAME):
        image = "public/VaccineInputUS/"+item['Image_Name']['S']
        print(item['id']['S']," ", item['Image_Name']['S']," ", item['Form_Name']['S'])
        df = create_final_df(image) 
        # if valid image, add to output table
        if(len(df)!=0):
            addOutputToTable(df,item)

    # CHECKING OUTPUT TABLE
    print("\nOutput Table")
    for item in scan_table(dynamo_client = dynamoDb_client, TableName = "VaccineOutputUS-a7zevy56jbg5vhz7otjz2zgu6i-staging"):
        # print(item)
        print(item['id']['S']," ", item['Form_Name']['S']," ",item['First_Name']['S']," ",item["Last_Name"]['S']," ", item["Last_Name"]['S'], item['Confidence_Level']['N'])




###### OLD CODE #######

    # Run your vaccine card
    # final_df = create_final_df(vaccineCardFile)

    # Run entire bucket of vaccine cards
    # s3 = boto3.resource('s3')
    # my_bucket = s3.Bucket(s3BucketVaccineCards)

    # final_df  = pd.DataFrame()
    # for my_bucket_object in my_bucket.objects.all():
    #     if (my_bucket_object.key != 'IMG_8541.jpg' and my_bucket_object.key != 'IMG_2708.jpg' and my_bucket_object.key != vaccineCardFile ):
    #         final_df= final_df.append(create_final_df(my_bucket_object.key),ignore_index=True)

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # final_df.to_csv("final_output.csv",index = False)
    

run()









from doctest import DocFileCase
from enum import auto
import boto3
import time
import json
from cv2 import dft
import pandas as pd
import numpy as np
from trp import Document


import re
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

s3 = boto3.client("s3")
textract = boto3.client("textract")

s3BucketVaccineCards = "demovaccinecards2022"  # REPLACE WITH S3 BUCKET NAME
vaccineCardFile = "Covid_Vaccine_Card.jpg"             # REPLACE FOR CARD IN BUCKET

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


    #first name, last name, dob, dose1manufacture, dose1date, dose1location,dose2manufacturer
    # FIX LATER 
    for page in doc.pages:
        for table in page.tables:
            data = [[cell.text for cell in row.cells] for row in table.rows]
            df = pd.DataFrame(data)
    
    # print(df)
    #df.to_csv('sample.csv')
    return df

def runTableAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile):

    textractJobId = start_analyze(s3BucketVaccineCards, vaccineCardFile, "TABLES")
    print("Started job with id: {}".format(textractJobId))
    
    if(CheckAnalyzeJobComplete(textractJobId)):
        df = get_textract_tables(textractJobId)

    return df

#runTableAnalyzeTextract()

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
            #"key_confidence": blocks[k]["Confidence"],
            #"value_id": v,
            "value_text": value_text,
            #"value_confidence": blocks[v]["Confidence"],
        }

        results.append(result)

    df = pd.DataFrame(results)
    df = df[df.value_text != '']
    #df[["key_text", "key_confidence", "value_text", "value_confidence"]].head()
    # print(df)
    return df


def runFormAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile):

    textractJobId = start_analyze(s3BucketVaccineCards, vaccineCardFile, "FORMS")
    print("Started job with id: {}".format(textractJobId))
    
    if(CheckAnalyzeJobComplete(textractJobId)):
        pages = get_textract_results(textractJobId)

    blocks = {block["Id"]: block for page in pages for block in page["Blocks"]}
    return get_form_dataframe(blocks)
    
    #print(pages[0])
    #with open('test.json', 'w') as json_file:
        #json.dump(pages[0], json_file)

#runFormAnalyzeTextract()

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

#Works only for tables
def correct_all_table(df):
    correct_words = {'pfizer':'vaccine$ pfizer', 'pfizer xxxxxx':'vaccine$ pfizer', 'pfizer-biontech': 'vaccine$ pfizer', 
        'moderna':'vaccine$ moderna', '1st dose':'dose1', '1st dose covid-19': 'dose1', '2nd Dose': 'dose2', 
        '2nd dose covid-19': 'dose2', 'walgreens': 'walgreens', 'date': 'Date Header',
        'product name/manufacturer lot number': 'Manufacturer Header', 'vaccine': 'Vaccine Header',
        'healthcare professional or clinic site': 'Site Header', 'other': 'none', 'mm dd yy': 'none'}
        #add more to correct_words

    #print(df)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            # https://www.stackvidhya.com/get-value-of-cell-from-a-pandas-dataframe/
            if type(df.iat[row,col]) == str:
                df.iat[row,col] = autocorrect(df.iat[row,col], correct_words, False)
    #print(df)
    #df.to_csv('sample2.csv')
    return df


def create_final_df(vaccine_card):
    form_df = runFormAnalyzeTextract(s3BucketVaccineCards, vaccine_card)
    form_df.set_index('key_text', inplace=True)
    #print(form_df)
    fields = ['First Name','Last Name','Date of birth']
    f_df = pd.DataFrame()
    for i in fields:
        if (form_df.empty) or (i not in form_df.index):
            f_df[i] = "N/A"
        else:
            f_df[i] = form_df.loc[i]

    table_df = runTableAnalyzeTextract(s3BucketVaccineCards, vaccine_card)
    corrected_df = correct_all_table(table_df)
    new_header = corrected_df.iloc[0] #grab the first row for the header
    corrected_df = corrected_df[1:] #take the data less the header row
    corrected_df.columns = new_header #set the header row as the df header

    #created separate dfs for the doses because they need to have different column names in the final df
    dose1_df = pd.DataFrame()
    dose2_df = pd.DataFrame()
    # print(corrected_df.get('Vaccine Header'))
    if 'Vaccine Header' in corrected_df.columns:
        dose1_df = corrected_df.loc[corrected_df['Vaccine Header'] == 'dose1']
        dose2_df = corrected_df.loc[corrected_df['Vaccine Header'] == 'dose2']
        
    else:
        # print(corrected_df.iloc[0])
        dose1_df = dose1_df.append(corrected_df.iloc[0])
        dose2_df = dose2_df.append(corrected_df.iloc[1])
    dose1_df = dose1_df.rename(columns={'Manufacturer Header': 'dose1_manufacturer', 'Date Header': 'dose1_date', 'Site Header': 'dose1_location' })
    dose2_df = dose2_df.rename(columns={'Manufacturer Header': 'dose2_manufacturer', 'Date Header': 'dose2_date', 'Site Header': 'dose2_location' })
    
    frames = [dose1_df, dose2_df]
    t_df = pd.concat(frames)
    t_df = t_df.fillna(method='bfill')
    t_df = t_df[:-1]

    #final_df contains form AND table dfs
    final_frames = [f_df, t_df]
    final_df = pd.concat(final_frames)
    final_df = final_df.fillna(method='bfill')
    final_df = final_df[:-1]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(final_df)
    return final_df

def run():
    # final_df = create_final_df("IMG_8541.jpg")
    # # final_df = create_final_df(vaccineCardFile)
    # print(final_df)

    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket("demovaccinecards2022")

    final_df  = pd.DataFrame()
    for my_bucket_object in my_bucket.objects.all():
        if (my_bucket_object.key != 'IMG_8541.jpg'):
            final_df= final_df.append(create_final_df(my_bucket_object.key),ignore_index=True)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    final_df.to_csv("final_output.csv",index = False)




   ### ANALYZE SINGLE VACCINE CARD
    # form_df = runFormAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile)

    # form_df.set_index('key_text', inplace=True)
    # print(form_df)
    # fields = ['First Name','Last Name','Date of birth']
    # df = pd.DataFrame()
    # for i in fields:
    #     df[i] = form_df.loc[i]
    # print(df)
 
    # table_df = runTableAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile)
    # corrected_df = correct_all_table(table_df)
    # print(table_df)
    # print(corrected_df)

    # field = ['dose1','dose2']
    # t_df = pd.DataFrame()
    # # corrected_df = corrected_df.iloc[1: , :]
    # # corrected_df.columns = corrected_df.iloc[0]
    # corrected_df.reset_index()
    # corrected_df.set_index(0, inplace=True)

    # # for col in corrected_df.columns:
    # #     print(col)
    # corrected_df = corrected_df.iloc[1: , :]
    # for i in field:
        
    #     temp = corrected_df.loc[i]
    #     # print(type(temp))
    #     # temp.resetIndex()
    #     print(temp)
    #     ind = 1
    #     t_df[ind] =i
    #     for j in temp:
    #     # print(type(temp[0]))
    #         t_df[ind] = j
    #         print(j)
    #         ind+=1
    
    # print(t_df)
    # # print(finaldf)
    

    ###RUN ALL FILES IN BUCKET AND OUTPUT INTO CSV IN FORM + TABLE FORMAT
    # s3 = boto3.resource('s3')
    # my_bucket = s3.Bucket('demovaccinecards2022')
    # form_df = {}
    # df = {}
    # corrected_df={}
    # for my_bucket_object in my_bucket.objects.all():
    #     if (my_bucket_object.key != 'IMG_8541.jpg'):
    #         print(my_bucket_object.key)
    #         form_df[my_bucket_object.key] = runFormAnalyzeTextract(s3BucketVaccineCards,  my_bucket_object.key)
    #         form_df[my_bucket_object.key] = form_df[my_bucket_object.key][form_df[my_bucket_object.key]['key_text'].str.contains('First Name|Last Name|Date of birth')]
    #         df[my_bucket_object.key] = runTableAnalyzeTextract(s3BucketVaccineCards, my_bucket_object.key)
    #         corrected_df[my_bucket_object.key] = correct_all_table(df[my_bucket_object.key])

    # pd.set_option("display.max_rows", None, "display.max_columns", None) #Added to print entire dataframe
    # print(form_df)
    # print(df)

    # with open('form_output.csv','w+') as f:
    #     for i in df:
    #         f.write("\n")
    #         f.write(i + "\n")
    #         form_df[i].to_csv(f,header = False, index = False)
    #         f.write("\n")
    #         df[i].to_csv(f,index = False)
    #         f.write("\n")
    
    # print("Auto-Corrected table")

    # print(corrected_df)

    # with open('corrected_form_output.csv','w+') as f:
    #     for i in corrected_df:
    #         f.write("\n")
    #         f.write(i + "\n")
    #         form_df[i].to_csv(f,header = False, index = False)
    #         f.write("\n")
    #         corrected_df[i].to_csv(f,index = False)
    #         f.write("\n")

    # #### prints all files in s3
    # s3 = boto3.resource('s3')
    # my_bucket = s3.Bucket('demovaccinecards2022')
    # for my_bucket_object in my_bucket.objects.all():
    #     print(my_bucket_object.key)

    # corrected_df.to_csv('sample2.csv')


   
    

run()






# arr = df.to_numpy()
# for row in range(arr.shape[0]):
#    for col in range(arr.shape[1]):
#        if type(arr[row, col]) == str:
#            arr[row, col] = autocorrect(arr[row, col])
# print(arr)


incorrect_words = ['HizeR', 'Ptizer', 'modna', 'HizeR EW0198', 'wapreeds', 'mg 4355', '1st Dose COVID-19 ']

correct_words = ['pfizer', 'moderna', 'walgreens']




#                    0                                      1                  2                                        3
#0            Vaccine   Product Name/Manufacturer Lot Number               Date   Healthcare Professional or Clinic Site 
#1  1st Dose COVID-19                           HizeR EW0198   7/10/21 mm dd yy                                 wapreeds 
#2           2nd Dose                                 Ptizer             8/2/21                                          
#3           COVID-19                                 FAT484           mm dd yy                                  mg 4355 
#4              Other                                              / / mm dd yy                                          
#5              Other                                              / / mm dd yy                                          
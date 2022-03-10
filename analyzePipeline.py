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

s3BucketVaccineCards = "demovaccinecardsjh"  # REPLACE WITH S3 BUCKET NAME
vaccineCardFile = "vaccine card JH.JPG"             # REPLACE FOR CARD IN BUCKET

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
    #we want all the values in a row, separated by commas
    #first name, last name, DOB, vaccine 1 (eg pfizer), vaccine 1 date, vaccine 1 location
    # FIX LATER 
    for page in doc.pages:
        for table in page.tables:
            data = [[cell.text for cell in row.cells] for row in table.rows]
            df = pd.DataFrame(data)
    
    #print(df)
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
    print(df)
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

def correct_all_table(df):
    correct_words = {'pfizer':'vaccine$ pfizer', 'pfizer xxxxxx':'vaccine$ pfizer', 'pfizer-biontech': 'vaccine$ pfizer', 
        'moderna':'vaccine$ moderna', '1st dose':'dose1', '1st dose covid-19': 'dose1', '2nd Dose': 'dose2', 
        '2nd dose covid-19': 'dose2', 'walgreens': 'walgreens', 'date': 'Date Header',
        'product name/manufacturer lot number': 'Manufacturer Header', 'vaccine': 'Vaccine Header',
        'healthcare professional or clinic site': 'Site Header', 'other': 'none', 'mm dd yy': 'none'}

    #print(df)
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            # https://www.stackvidhya.com/get-value-of-cell-from-a-pandas-dataframe/
            if type(df.iat[row,col]) == str:
                df.iat[row,col] = autocorrect(df.iat[row,col], correct_words, False)
    #print(df)
    #df.to_csv('sample2.csv')
    return df

def get_doses_as_string(corrected_df):
    dose1_df = corrected_df[corrected_df[0].str.contains('dose1')]
    dose1_list = dose1_df.astype(str).values.flatten().tolist()
    if len(dose1_list) > 0:
        dose1_list.pop(0) #removes 'dose1' element from list

    dose2_df = corrected_df[corrected_df[0].str.contains('dose2')]
    dose2_list = dose2_df.astype(str).values.flatten().tolist()
    if len(dose2_list) > 0:
        dose2_list.pop(0) #removes 'dose2' element from list

    #print(dose2_list)
    dose1_list.extend(dose2_list)
    doses_list = dose1_list #combining this line with the line above it doesn't work somehow
    dose_info = "" #comma separated string of dose 1 and 2 info ONLY
    for i in range(len(doses_list)):
        dose_info += doses_list[i] + ","
    dose_info = dose_info[0:len(dose_info)-1]
    return dose_info

def create_final_df(vaccine_card):
    form_df = runFormAnalyzeTextract(s3BucketVaccineCards, vaccine_card)
    form_df.set_index('key_text', inplace=True)
    #print(form_df)
    fields = ['First Name','Last Name','Date of birth']
    f_df = pd.DataFrame()
    for i in fields:
        if i in form_df.index:
            f_df[i] = form_df.loc[i]
        else:
            f_df[i] = 'N/A'

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
    final_df = final_df.fillna(method='bfill')
    final_df = final_df[:-1]
    final_df = final_df.replace('', "N/A")
    final_df = final_df.replace(np.nan, "N/A")
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(final_df)
    return final_df

def run():
    #df = runTableAnalyzeTextract()
    #df.head()
    #FORM STUFF
    # df = df[df['key_text'].str.contains('Last Name')]
    # print(df['value_text'])
    #print(df)
    #END FORM STUFF
    # print(t_df)
    # print()
    # df = create_final_df("RedacteUSAvax4.png")
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(df)
    # print(df['dose1_location'].values)
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(s3BucketVaccineCards)

    final_df  = pd.DataFrame()
    for my_bucket_object in my_bucket.objects.all():
        if (my_bucket_object.key != 'IMG_8541.jpg'):
            final_df= final_df.append(create_final_df(my_bucket_object.key),ignore_index=True)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    final_df.to_csv("final_output.csv",index = False)

    

    # s3 = boto3.resource('s3')
    # my_bucket = s3.Bucket(s3BucketVaccineCards)

    # final_df  = pd.DataFrame()
    # for my_bucket_object in my_bucket.objects.all():
    #     if (my_bucket_object.key != 'IMG_8541.jpg' and my_bucket_object.key != 'IMG_5027.jpeg'):
    #         final_df= final_df.append(create_final_df(my_bucket_object.key),ignore_index=True)

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # final_df.to_csv("final_output.csv",index = False)
    

    # df = runTableAnalyzeTextract(s3BucketVaccineCards, vaccineCardFile)
    # corrected_df = correct_all_table(df)
    # print(get_doses_as_string(corrected_df))
    #df = df[df['key_text'].str.contains('Last Name')]
    #df = pd.read_csv('sample.csv')
    #print(df)
    
    #corrected_df = pd.read_csv('sample2.csv')
    
    #print(dose1_df)

    #corrected_df.to_csv('sample2.csv')
    # s3 = boto3.resource('s3')
    # my_bucket = s3.Bucket("demovaccinecardsjh")
    # df = {}
    # corrected_df={}
    # for my_bucket_object in my_bucket.objects.all():
    #     df[my_bucket_object.key] = runTableAnalyzeTextract(s3BucketVaccineCards, my_bucket_object.key)
    #     corrected_df[my_bucket_object.key] = correct_all_table(df[my_bucket_object.key])
    # pd.set_option("display.max_rows", None, "display.max_columns", None) 
    # #print(df)
    # #print(corrected_df)
    # with open('doses_output.csv','w+') as f:
    #     for i in corrected_df:
    #         f.write("\n")
    #         f.write(i)
    #         f.write("\n")
    #         f.write(get_doses_as_string(corrected_df[i]))
    #         #corrected_df[i].to_csv(f,index = False)
    #         f.write("\n")
    #stores as csv
    # with open('output.csv','w+') as f:
    #     for i in df:
    #         f.write("\n")
    #         f.write(i)
    #         df[i].to_csv(f,index = False)
    #         f.write("\n")
    
    #print("Auto-Corrected table")
    # corrected_df = correct_all_table(df)
    # # corrected_df = pd.read_csv('sample2.csv')
    # corrected_df.to_csv("corrected_tables.csv", index = False)
    # print(corrected_df)
    # with open('corrected_output.csv','w+') as f:
    #     for i in corrected_df:
    #         f.write("\n")
    #         f.write(i)
    #         corrected_df[i].to_csv(f,index = False)
    #         f.write("\n")
    
    

run()








#arr = df.to_numpy()
#for row in range(arr.shape[0]):
#    for col in range(arr.shape[1]):
#        if type(arr[row, col]) == str:
#            arr[row, col] = autocorrect(arr[row, col])
#print(arr)


incorrect_words = ['HizeR', 'Ptizer', 'modna', 'HizeR EW0198', 'wapreeds', 'mg 4355', '1st Dose COVID-19 ']

correct_words = ['pfizer', 'moderna', 'walgreens']




#                    0                                      1                  2                                        3
#0            Vaccine   Product Name/Manufacturer Lot Number               Date   Healthcare Professional or Clinic Site 
#1  1st Dose COVID-19                           HizeR EW0198   7/10/21 mm dd yy                                 wapreeds 
#2           2nd Dose                                 Ptizer             8/2/21                                          
#3           COVID-19                                 FAT484           mm dd yy                                  mg 4355 
#4              Other                                              / / mm dd yy                                          
#5              Other                                              / / mm dd yy                                          
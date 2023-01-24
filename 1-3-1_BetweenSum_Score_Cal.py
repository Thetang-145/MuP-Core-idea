import sys
import json
import pandas as pd
from datasets import load_metric
from evaluate import load

# Matric score download
rougescore = load_metric("rouge")
bertscore = load("bertscore")

# Load dataset
def load_data(dts):
    path = f'MuP_dataset/{dts}_complete.jsonl'
    try:
        with open(path, 'r') as json_file:
            json_list = list(json_file)
        col_name = ["paper_id","summary"]
    except:
        print(f"Warning: Did not load dataset from {path}")
        return
    summary_df = pd.DataFrame(columns=col_name)
    for json_str in json_list[:]:
        result = json.loads(json_str)
        df = pd.DataFrame([[result["paper_id"], result["summary"]]], columns=col_name)
        summary_df = pd.concat([summary_df,df])
    return summary_df

def split_sum_num(df):
    num_paper = df.groupby(['paper_id']).count()
    num_paper['num_paper'] = 1
    num_paper.groupby(['summary']).count()
    num_paper.drop('num_paper', inplace=True,axis=1)
    num_paper.sort_values(["summary"])
    df_list = []
    for i in range(0, max(num_paper['summary'])):
        try:
            paper_id = (list((num_paper[num_paper["summary"]==(i+1)]).index))
            df_i = df[df.paper_id.isin(paper_id)].sort_values("paper_id")
            df_list.append(df_i.groupby('paper_id').apply(lambda df_: df_[['summary']].values.flatten()).apply(pd.Series).reset_index())
        except:
            df_list.append(None)
    return df_list

# Function: Generate score dataframe
def n_scores(df, subscore_col):
    n = len(df.columns)-1
    pairs = [f'{i}-{j}' for i in range(n) for j in range(i+1, n)]
    col = pd.MultiIndex.from_product([subscore_col, pairs])
    scores = pd.DataFrame(columns=col)
    scores.insert(0, "paper_id", df["paper_id"])
    # scores.set_index("paper_id", inplace=True)
    return scores

# Function: Rouge score calculation
def rouge_cal(df):
    n = len(df.columns)-1
    print(f"\nCalculating ROUGE on {n} summaries")

    rouge_list = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    score_list = ['precision', 'recall', 'fmeasure']

    df_score_dict = {}
    mux = pd.MultiIndex.from_product([["summary"],(list(df.columns))[1:]])
    df_score = pd.DataFrame(columns=mux)
    df_score.insert(0, "paper_id", df["paper_id"])
    for col in df:
        if col != 'paper_id':
            df_score[('summary', col)] = df[col]
    df_score = df_score.merge(n_scores(df, score_list), left_on='paper_id', right_on='paper_id')
    for r in rouge_list:
        df_score_dict[r] = df_score

    df_len = len(df)
    for idx, row in df.iterrows():
        sys.stdout.write(f"\r{idx+1}/{df_len}")
        sys.stdout.flush()
        for i in range(n):
            for j in range(i+1,n):
                pair = f'{i}-{j}'
                score = rougescore.compute(predictions=[row[i]], references=[row[j]], use_stemmer=False)
                for r in rouge_list:
                    df_score_dict[r].loc[idx, ('precision', pair)] = ((score[r]).low).precision
                    df_score_dict[r].loc[idx, ('recall', pair)] = ((score[r]).low).recall
                    df_score_dict[r].loc[idx, ('fmeasure', pair)] = ((score[r]).low).fmeasure

    return df_score_dict

# Function: BERTScore calculation
def bertscore_cal(df):
    n = len(df.columns)-1
    print(f"Calculating BERTScore on {n} summaries")
    
    score_list = ['precision', 'recall', 'f1']

    mux = pd.MultiIndex.from_product([["summary"],(list(df.columns))[1:]])
    df_score = pd.DataFrame(columns=mux)
    df_score.insert(0, "paper_id", df["paper_id"])
    for col in df:
        if col != 'paper_id':
            df_score[('summary', col)] = df[col]
    df_score = df_score.merge(n_scores(df, score_list), left_on='paper_id', right_on='paper_id')

    for i in range(n):
        for j in range(i+1,n-1):
            pair = f'{i}-{j}'
            summary1 = list(df_score.loc[:, ('summary', i)])
            summary2 = list(df_score.loc[:, ('summary', j)])
            print(pair)
            result = bertscore.compute(predictions=summary1, references=summary2, lang="en", rescale_with_baseline=True)
            for score in score_list:
                df_score.loc[:, (score, pair)] = result[score]

    return df_score
    

def main(dts="training", cal_rouge=True, cal_bert=True):
    summary_df = load_data(dts)
    df_list = split_sum_num(summary_df)

    # ROUGE score calculation and Save
    if cal_rouge:
        for n, df in enumerate(df_list):
            if n+1 > 1:
                if df==None: continue
                dict_result = rouge_cal(df)
                for key, val in dict_result.items():
                    val.to_csv(f"visualization_data/rouge-between-sum/{key}/{dts}_{key}_{n+1}sum.csv")

    # BERTScore calculation and Save
    if cal_bert:
        for n, df in enumerate(df_list):
            if n+1 > 1:
                if df==None: continue
                result = bertscore_cal(df)
                val.to_csv(f"visualization_data/bertscore-between-sum/{dts}_bertscore_{n+1}sum.csv")

if __name__ == '__main__':
    main(dts="validation", cal_bert=False)
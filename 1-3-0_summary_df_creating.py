import sys
import json
import pandas as pd
from datasets import load_metric

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

def main(dts):
    df = load_data(dts)
    num_paper = df.groupby(['paper_id']).count()
    num_paper['num_paper'] = 1
    num_paper.groupby(['summary']).count()
    num_paper.drop('num_paper', inplace=True,axis=1)
    num_paper.sort_values(["summary"])

    list_paper_id = []
    for i in range(1, max(num_paper['summary'])+1):
        list_paper_id.append(list((num_paper[num_paper["summary"]==i]).index))
    for id in list_paper_id:
        print(len(id))

    df_list = []
    for i in range(len(list_paper_id)):
        df_list.append(summary_df[summary_df.paper_id.isin(list_paper_id[i])].sort_values("paper_id"))
    df_list[1][:20]

if __name__ == '__main__':
    main("training")
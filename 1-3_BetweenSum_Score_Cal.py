import sys
import json
import pandas as pd
from evaluate import load

bertscore = load("bertscore")

def n_scores(n, list_paper_id, score_col=["Bert"]):
    pairs = []
    for i in range(n):
        for j in range (i+1,n):
            pairs.append(str(i+1)+"-"+str(j+1))
    col = pd.MultiIndex.from_product([score_col, pairs])
    scores = pd.DataFrame(columns=col)
    scores.insert(0, "paper_id", list_paper_id[n-1])
    scores.set_index("paper_id", inplace=True)
    return scores

def bertscore_cal(n, list_paper_id, df_list):
    df_scores = n_scores(n, list_paper_id, ["Bert"])
    for num_id,id in enumerate(list_paper_id[n-1]):
        sumaries = list(df_list[n-1][df_list[n-1]["paper_id"]==id]["summary"])
        sys.stdout.write("\r" + str(num_id) + "/" + str(len(list_paper_id[n-1])))
        sys.stdout.flush()
        # print(id,"/",len(list_paper_id[n-1]), flush=True)
        for i in range(n):
            for j in range(i+1,n):
                score = bertscore.compute(predictions=[sumaries[i]], references=[sumaries[j]], lang="en")

                pair = str(i+1)+"-"+str(j+1)
                df_scores.loc[id]['Bert'][pair] = score['f1'][0]
        #         print(score)
        #         print(score['f1'][0])
        #         break
        #     break
        # break
    # df_score2
    # len(list_paper_id)
    print("\n")
    return df_scores

def main():

    dts = "training"
    # dts = "validation"
    path = 'MuP_dataset/'+dts+'_complete.jsonl'
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    col_name = ["paper_id","summary"]
    summary_df = pd.DataFrame(columns=col_name)
    for json_str in json_list:
        result = json.loads(json_str)
        df = pd.DataFrame([[result["paper_id"], result["summary"]]], columns=col_name)
        summary_df = pd.concat([summary_df,df])

    num_paper = summary_df.groupby(['paper_id']).count()
    num_paper['num_paper'] = 1

    num_paper.drop('num_paper', inplace=True,axis=1)
    num_paper.sort_values(["summary"])

    list_paper_id = []
    for i in range(1, max(num_paper['summary'])+1):
        list_paper_id.append(list((num_paper[num_paper["summary"]==i]).index))

    df_list = []
    for i in range(len(list_paper_id)):
        df_list.append(summary_df[summary_df.paper_id.isin(list_paper_id[i])].sort_values("paper_id"))

    for n in range(2,len(list_paper_id)):
        print("Calculating ROUGE on", n, "summaries")
        df_scores = bertscore_cal(n, list_paper_id, df_list)

        df_scores.to_csv("visualization_data/BertScore-between-sum/"+dts+"_bertscore_"+str(n)+"sum.csv")
        # break

if __name__ == '__main__':
    main()
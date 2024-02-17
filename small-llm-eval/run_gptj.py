import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import pandas as pd

def main(locationGpt2HF,outputFileName,inputFileName):
    tokenizer = AutoTokenizer.from_pretrained(locationGpt2HF)
    model = AutoModelForCausalLM.from_pretrained(locationGpt2HF)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## read prompts from csv and generate predictions
    df=pd.read_csv(inputFileName)
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['pred']
        writer.writerow(row)

    with torch.no_grad():
        for i in range(0,df.shape[0],1):
            prompts=list(df['text'].values)[i:i+1]
            #get the type of the list element
            print(type(prompts))
            print(prompts[0])
            prompts=prompts[0]+"\nAnswer:"
            inputs = tokenizer([prompts], return_tensors='pt').to(device)

            output_sequences = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                do_sample=False,
                max_new_tokens=10,temperature=0
            )
            outputs=tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            outputs=[[el] for el in outputs]
            with open(outputFileName, 'a') as csvoutput:
                writer = csv.writer(csvoutput, lineterminator='\n')
                writer.writerows(outputs)

if __name__ == '__main__':
    ## model location HuggingFace format
    locationGpt2HF="EleutherAI/gpt-j-6B"
    outputFileName="./gptj_causal_benchmark.csv"
    inputFileName="./prompts.csv"
    main(locationGpt2HF,outputFileName,inputFileName)
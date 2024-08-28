from transformers import AutoTokenizer,TokenClassificationPipeline,AutoModelForTokenClassification
from fuzzywuzzy import process
import pandas as pd


def fuzzy_match(input_string, options_list):
    best_match, similarity_score = process.extractOne(input_string, options_list)
    threshold = 80
    if similarity_score >= threshold:
        return best_match
def read_database_drugs(filename="database.txt"):
    with open(filename, "r") as f:
        out = f.readlines()
    return out
def main(out, lists):
    if out:
        w = out[0]["word"]
        c = fuzzy_match(w, lists)
        if c:
            return c
        return w

if __name__ == "__main__":
    with open("input.txt", "r") as f:
        input_text = f.readlines()
    database = read_database_drugs()
    model = AutoModelForTokenClassification.from_pretrained("./out")
    tokenizer = AutoTokenizer.from_pretrained("./out")
    predict = TokenClassificationPipeline(model= model,tokenizer=tokenizer, framework='pt', task='ner', aggregation_strategy="simple")
    out = predict(input_text[0])
    c = main(out, database)
    print(f"Correct drug's name : {c}")
    with open("output_22211262.txt", "w") as f:
        if c:
            f.writelines(c)
        else:
            f.writelines("")

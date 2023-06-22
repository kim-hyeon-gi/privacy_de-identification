import argparse
from glob import iglob



def main():

    f = open("/home/hyeongikim/Desktop/kobert_eagle/pytorch-bert-crf-ner/data_in/CN_Dataset/suggestion/suggestion_data.txt","r")
    t = open("/home/hyeongikim/Desktop/kobert_eagle/pytorch-bert-crf-ner/data_in/CN_Dataset/suggestion/suggestion_raw_data.txt","w")
    lines = f.readlines()
    for line in lines:
        line = line.replace(".",".\n")

        print(line, file=t)


if __name__ == '__main__':

    main()

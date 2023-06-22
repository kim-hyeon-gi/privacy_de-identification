import codecs


def load_data_from_txt(file_full_name = '/home/hyeongikim/Desktop/kobert_eagle/pytorch-bert-crf-ner/data_in/NER-master/말뭉치 - 형태소_개체명/00002_NER.txt'):
        with codecs.open(file_full_name, "r", "utf-8") as io:
            lines = io.readlines()

            # parsing에 문제가 있어서 아래 3개 변수 도입!
            prev_line = ""
            save_flag = False
            count = 0
            sharp_lines = []

            for line in lines:
                if prev_line == "\n" or prev_line == "":
                    save_flag = True
                if line[:3] == "## " and save_flag is True:
                    count += 1
                    sharp_lines.append(line[3:])
                if count == 3:
                    count = 0
                    save_flag = False

                prev_line = line
            list_of_source_no, list_of_source_str, list_of_target_str = sharp_lines[0::3], sharp_lines[1::3], sharp_lines[2::3]
        return list_of_source_no, list_of_source_str, list_of_target_str

print(load_data_from_txt())
    

import os
import extract_v3

os.chdir("D:\\TEST\\ml_graphology_IP\\OGimages")
files = [f for f in os.listdir('.') if os.path.isfile(f)]
# os.chdir("..")
print(files)
#working

page_ids = []
with open("D:\\TEST\\ml_graphology_IP\\raw_feature_list.txt", "r") as label:
    for line in label:
        content = line.split()
        page_id = content[-1]
        page_ids.append(page_id)

with open("D:\\TEST\\ml_graphology_IP\\raw_feature_list.txt", "a") as label:
    count = len(page_ids)
    for file_name in files:
        if file_name in page_ids:
            continue
        features = extract_v3.start(file_name)
        features.append(file_name)
        for i in features:
            label.write("%s\t" % i)
        print(label, '')
        count += 1
        progress = (count*100)/len(files)
        print(str(count)+' '+file_name+' '+str(progress)+'%')
    print("Done!")

import os
from PIL import Image
import random
from random import seed
from random import randint
import json
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator
from detect_from_video import predict_with_model
import cv2
import torch
print("TORCH CUDA:" + str(torch.cuda.is_available()))
import FaceMorpherNoBG as fm
import face_recognition
from tqdm import tqdm


import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

global family_tree
global objects_array
global obj_stats
global obj_acc
global obj_total




seed(1)

# TODO: set evaluation metric to pick best images
def df_column_uniquify(df):
    df_columns = df.columns
    new_columns = []
    for item in df_columns:
        counter = 0
        newitem = item
        while newitem in new_columns:
            counter += 1
            newitem = "{}_{}".format(item, counter)
        new_columns.append(newitem)
    df.columns = new_columns
    return df

class GeneticAlgorithmFaceMerge:
    # Class Constructor with with all the arguments
    # base_images_path: path to the template images
    # merge_images_path: path to the images to be merged
    # output_path: path to the output folder
    # max_generations: number of maximum generations
    # max_images: maximum number of images per generation
    # m: probabilty of mutation occurring
    # requests_sleep_time: time to sleep between requests
    def __init__(self, before_images_path,base_images_path, merge_images_path, output_path,model,feature_rate=75, max_generations=1000, max_images=1000, m=0.01,flip_images=False,contrast=0,start_conf=80):
        self.faceplusplus_url = "https://api-us.faceplusplus.com/imagepp/v1/mergeface"
        self.api_key = "58Gz-OF87DNTbKD9QhHCwOu0pB6Kd1zo"
        self.api_secret = "i4xoBKEmDJABa2J1S37PXJa19X-UKUla"
        self.model=model
        self.feature_rate=feature_rate
        self.mutation_rate = m
        self.crossover_rate = 1 - m
        self.base_images_path = base_images_path
        self.merge_images_path = merge_images_path
        self.before_images_path = before_images_path
        self.base_images = os.listdir(self.base_images_path)
        self.merge_images = os.listdir(self.merge_images_path)
        print(self.before_images_path)
        self.before_images = os.listdir(self.before_images_path)
        self.output_folder = output_path
        self.flip_images=flip_images
        self.contrast=contrast
        self.max_generations = max_generations
        self.max_images_per_generation = max_images
        self.num_generated_per_gen = 0
        self.generation = 0
        self.fitness = 0
        self.generated_count = 0
        self.number_mutations_occured = 0
        self.number_crossover_occured = 0
        self.next_child_dir_path = ""
        self.previous_child_dir_path = ""
        self.yes=0
        self.no=0
        self.real_confidence_val=start_conf
        self.start_real_confidence_step=1
        self.start_real_confidence_max=100




    def change_contrast(self,img, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))
        def contrast(c):
            return 128 + factor * (c - 128)
        return img.point(contrast)


    def faceMerge(self, base_face, merge_face, save_path,flip=False,contrast=0,alpha=0.5):
        
        Image2=fm.merge2images(base_face,merge_face,detector,predictor,alpha)
        Image3 = Image.fromarray(Image2)
        # Image3 = Image3.convert('RGBA').tobytes('raw', 'RGBA')
        if(flip):
            if(random.randint(0,1)==0):
                Image3 = Image3.transpose(method=Image.FLIP_LEFT_RIGHT)
        Image3=self.change_contrast(Image3,contrast)
        Image3.save(save_path, 'JPEG')
        # time.sleep(self.requests_sleep_time)

    # arrays of images are shuffled in each step to prevent repetition of same base image when max_images_per_generation is lower than total images

    def GeneticAlgorithmMerge(self):
        random.shuffle(self.base_images)
        random.shuffle(self.merge_images)
        for iter in range(self.max_generations):
            self.num_generated_per_gen = 0
            if(self.generation < self.max_generations):
                print("----------------------")
                print("Generation :"+str(self.generation))
                self.Selection()
                print("----------------------")
                self.generation += 1
                if(self.generation < self.max_generations):
                    self.real_confidence_val+=self.start_real_confidence_step
            else:
                break

        family_tree["Tree"] = objects_array
        with open(self.output_folder+"/family_tree.json", 'w') as f:
            json.dump(family_tree, f)
        print("----------------------")
        print("Generation :"+str(self.generation))
        print("Number of Mutations occurred :" +
              str(self.number_mutations_occured))
        print("Number of Crossover occurred :" +
              str(self.number_crossover_occured))

        keys_values = obj_stats.items()


        keys_values_acc = obj_acc.values()


        keys_values_tot = obj_total.values()
        with open(self.output_folder+"/rejected.json", "w") as write_file:
            json.dump(obj_stats, write_file, indent=4)

        with open(self.output_folder+"/accepted.json", "w") as write_file:
            json.dump(obj_acc, write_file, indent=4)

        with open(self.output_folder+"/total.json", "w") as write_file:
            json.dump(obj_total, write_file, indent=4)

        df1=pd.DataFrame([obj_stats])
        df1=df1.transpose()
        df2=pd.DataFrame([keys_values_acc])
        df2=df2.transpose()
        df3=pd.DataFrame([keys_values_tot])
        df3=df3.transpose()
        df=pd.concat([df1,df2,df3],axis=1,join="inner")
        df=df_column_uniquify(df)
        df = df.rename(columns={df.columns[0]: 'Rejected'})
        df = df.rename(columns={df.columns[1]: 'Accepted'})
        df = df.rename(columns={df.columns[2]: 'Total'})
        df['Percentage']=""

        for index,row in df.iterrows():
            rej = df.iloc[index,0]
            tot = df.iloc[index,2]
            perc = int(rej)/int(tot)
            df.iloc[index,3]= perc*100

        
        # df=df.drop("Rejected",axis=1)
        # df=df.drop("Accepted",axis=1)
        # df=df.drop('Total',axis=1)
        print(df)

        ax=df['Percentage'].plot.line()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Generation")
        ax.set_ylabel("Percentage Rejected/Total")
        ax.get_figure().savefig(self.output_folder+'/stats.jpg')


        
        print("Done............")


    # Function used to select images for Crossover or mutation
    # If generation=0, takes images form base path and merge path
    # if generation>0, takes images from previous generation

    def Selection(self):
        operation = ['M', 'C']
        self.next_child_dir_path = self.output_folder + str(self.generation)
        if(self.generation == 0):
            array_images = []
            arr1=self.base_images
            arr2=self.merge_images
            random.shuffle(arr1)
            random.shuffle(arr2)
            for base in arr1:
                for merge in arr2:
                    if(merge != base):
                        before = os.path.split(merge)[0]+os.path.split(merge)[1].split(".")[0]+"_before.jpg"
                        array_images.append([base, merge,before])

            random.shuffle(array_images)
            for k in array_images:
                pick = random.choices(operation, weights=(
                    self.mutation_rate, self.crossover_rate), k=1)
                if(pick[0] == 'M'):
                    try:
                        self.Mutation(k[0], k[1],k[2])
                    except:
                        print("Error in Mutation")
                if(pick[0] == 'C'):
                    try:
                        self.Crossover(k[0], k[1],k[2])
                    except:
                        print("Error in Crossover")

        elif(self.generation==1):
            self.previous_child_dir_path = self.output_folder + \
                "/"+str(self.generation-1)
            self.next_child_dir_path = self.output_folder + \
                "/"+str(self.generation)
            if not(os.path.exists(self.next_child_dir_path)):
                os.mkdir(self.next_child_dir_path)

            merge_array = os.listdir(self.previous_child_dir_path)
            base_array = os.listdir(self.previous_child_dir_path)
            random.shuffle(merge_array)
            random.shuffle(base_array)

            array_images = []

            for base in base_array:
                for merge in merge_array:
                    if(merge != base):
                        name_split=os.path.split(merge)[1].split(".")[0]
                        before = os.path.split(merge)[0]+str(name_split.split("_")[0])+"_before_"+str(name_split.split("_")[1])+".jpg"
                        array_images.append([base, merge,before])

            

            random.shuffle(array_images)

            for k in array_images:

                pick = random.choices(operation, weights=(
                    self.mutation_rate, self.crossover_rate), k=1)
                if(pick[0] == 'M'):
                    try:
                        self.Mutation(k[0], k[1],k[2])
                    except:
                        print("Error in Mutation")
                if(pick[0] == 'C'):
                    try:
                        self.Crossover(k[0], k[1],k[2])
                    except:
                        print("Error in Crossover")

        elif(self.generation >= 2):
            self.previous_child_dir_path = self.output_folder + \
                "/"+str(self.generation-1)
            self.next_child_dir_path = self.output_folder + \
                "/"+str(self.generation)
            if not(os.path.exists(self.next_child_dir_path)):
                os.mkdir(self.next_child_dir_path)

            old_child=self.previous_child_dir_path.split("/")[0] + "/"+str(int(self.previous_child_dir_path.split("/")[1])-1)
            merge_array = os.listdir(self.previous_child_dir_path)
            base_array = os.listdir(old_child)
            random.shuffle(merge_array)
            random.shuffle(base_array)

            array_images = []

            for base in base_array:
                for merge in merge_array:
                    if(merge != base):
                        name_split=os.path.split(merge)[1].split(".")[0]
                        before = os.path.split(merge)[0]+str(name_split.split("_")[0])+"_before_"+str(name_split.split("_")[1])+".jpg"
                        array_images.append([base, merge,before])

            random.shuffle(array_images)

            for k in array_images:

                pick = random.choices(operation, weights=(
                    self.mutation_rate, self.crossover_rate), k=1)
                if(pick[0] == 'M'):
                    try:
                        self.Mutation(k[0], k[1],k[2])
                    except:
                        print("Error in Mutation")
                if(pick[0] == 'C'):
                    try:
                        self.Crossover(k[0], k[1],k[2])
                    except:
                        print("Error in Crossover")

        


    # Mutation Function
    # When a mutation occurrs, the value of merge_rate and feature_rate is selected randomly between 0 and 100 and then the two images are merged and saved into the child directory
    def Mutation(self, base, merge,before):
        if(self.num_generated_per_gen < self.max_images_per_generation):
            self.number_mutations_occured += 1
            feature_rate = randint(0, 100)
            merge_rate = randint(0, 100)
            if not(os.path.exists(self.output_folder+"/"+str(self.generation))):
                os.mkdir(self.output_folder+"/"+str(self.generation))
            if not(os.path.exists(self.output_folder+"/"+str(self.generation)+"_before")):
                os.mkdir(self.output_folder+"/"+str(self.generation)+"_before")
            if(self.generation == 0):
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                self.faceMerge(self.base_images_path+"/"+base, self.merge_images_path+"/"+merge, 
                               self.output_folder+"/"+str(self.generation)+"/" + str(self.generated_count) + "_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(self.base_images_path+"/"+base, self.before_images_path +"/"+before, 
                               self.output_folder+"/"+str(self.generation)+"_before/" + str(self.generated_count) + "_before_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                #image = Image.open(self.output_folder+"/"+str(self.generation)+"/" + str(self.generated_count) + "_mutation.jpg")
                imcv2=cv2.imread(self.output_folder+"/"+str(self.generation)+"/" + str(self.generated_count) + "_mutation.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Mutation',
                    'Current': str(self.generated_count) + "_mutation.jpg",
                    'Parent1': self.base_images_path+"/"+base,
                    'Parent2': self.merge_images_path+"/"+merge,
                    'Merge_Rate': merge_rate,
                    'Feature_Rate': feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before':str(self.generated_count) + "_before_mutation.jpg"
                }


                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1

                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Mutation")
                else:
                    
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.output_folder+"/"+str(self.generation) +
                                "/" + str(self.generated_count) + "_mutation.jpg")
                    os.remove(self.output_folder+"/"+str(self.generation)+"_before/" + str(self.generated_count) + "_before_mutation.jpg")


            elif(self.generation==1):
                base_split=base.split("_")
                base_before = base_split[0]+"_before_"+base_split[1]
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                self.faceMerge(self.previous_child_dir_path+"/"+base, self.previous_child_dir_path+"/"+merge,
                               self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(self.previous_child_dir_path+"_before/"+base_before, self.previous_child_dir_path+"_before/"+before,
                               self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                #image = Image.open(self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg")
                imcv2=cv2.imread(self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Mutation',
                    'Current': str(self.generated_count) + "_mutation.jpg",
                    'Parent1': self.previous_child_dir_path+"/"+base,
                    'Parent2': self.previous_child_dir_path+"/"+merge,
                    'Merge_Rate': merge_rate,
                    'Feature_Rate': feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before':str(self.generated_count) + "_before_mutation.jpg"
                }

                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1

                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Mutation")
                else:
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.output_folder+"/"+str(self.generation) +
                                "/" + str(self.generated_count) + "_mutation.jpg")
                    os.remove(self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_mutation.jpg")

            elif(self.generation>=2):
                base_split=base.split("_")
                base_before = base_split[0]+"_before_"+base_split[1]
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                old_child1=self.previous_child_dir_path.split("/")[0] + "/"+str(int(self.previous_child_dir_path.split("/")[1])-1)
                self.faceMerge(old_child1+"/"+base, self.previous_child_dir_path+"/"+merge, 
                               self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(old_child1+"_before/"+base_before, self.previous_child_dir_path+"_before/"+before, 
                               self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_mutation.jpg",self.flip_images,self.contrast,self.feature_rate)
                #image = Image.open(self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg")
                imcv2=cv2.imread(self.next_child_dir_path+"/" + str(self.generated_count) + "_mutation.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Mutation',
                    'Current': str(self.generated_count) + "_mutation.jpg",
                    'Parent1': old_child1+"/"+base,
                    'Parent2': self.previous_child_dir_path+"/"+merge,
                    'Feature_Rate': feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before': str(self.generated_count) + "_before_mutation.jpg"
                }

                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1
                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Mutation")
                else:
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.output_folder+"/"+str(self.generation) +
                                "/" + str(self.generated_count) + "_mutation.jpg")
                    os.remove(self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_mutation.jpg")



            objects_array.append(family)
            # self.generated_count += 1
            # self.num_generated_per_gen += 1
            print("Mutation")

    # Crossover Function
    # Crossover function function merges two images using faceplusplus api and saves the result in the child directory
    def Crossover(self, base, merge,before):
        if(self.num_generated_per_gen < self.max_images_per_generation):
            self.number_crossover_occured += 1
            if not(os.path.exists(self.output_folder+"/"+str(self.generation))):
                os.mkdir(self.output_folder+"/"+str(self.generation))
            if not(os.path.exists(self.output_folder+"/"+str(self.generation)+"_before")):
                os.mkdir(self.output_folder+"/"+str(self.generation)+"_before")
            if(self.generation == 0):
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                self.faceMerge(self.base_images_path+"/"+base, self.merge_images_path+"/"+merge,
                               self.output_folder+"/"+str(self.generation)+"/" + str(self.generated_count) + "_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(self.base_images_path+"/"+base, self.before_images_path+"/"+before,
                               self.output_folder+"/"+str(self.generation)+"_before/" + str(self.generated_count) + "_before_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                #image = Image.open(self.output_folder+"/"+str(self.generation) +"/" + str(self.generated_count) + "_crossover.jpg")
                imcv2=cv2.imread(self.output_folder+"/"+str(self.generation) +"/" + str(self.generated_count) + "_crossover.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Crossover',
                    'Current': str(self.generated_count) + "_crossover.jpg",
                    'Parent1': self.base_images_path+"/"+base,
                    'Parent2': self.merge_images_path+"/"+merge,
                    'Feature_Rate': self.feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before':str(self.generated_count) + "_before_crossover.jpg"
                }

                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1

                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Crossover")
                else:
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.output_folder+"/"+str(self.generation) +"/" + str(self.generated_count) + "_crossover.jpg")
                    os.remove(self.output_folder+"/"+str(self.generation)+"_before/" + str(self.generated_count) + "_before_crossover.jpg")

                        
            elif(self.generation==1):
                base_split=base.split("_")
                base_before = base_split[0]+"_before_"+base_split[1]
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                self.faceMerge(self.previous_child_dir_path+"/"+base, self.previous_child_dir_path+"/"+merge,
                                self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(self.previous_child_dir_path+"_before/"+base_before, self.previous_child_dir_path+"_before/"+before,
                                self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                # image = Image.open(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                imcv2=cv2.imread(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Crossover',
                    'Current': str(self.generated_count) + "_crossover.jpg",
                    'Parent1': self.previous_child_dir_path+"/"+base,
                    'Parent2': self.previous_child_dir_path+"/"+merge,
                    'Feature_Rate': self.feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before':str(self.generated_count) + "_before_crossover.jpg"
                }

                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1

                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Crossover")
                else:
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                    os.remove(self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_crossover.jpg")

            elif(self.generation>=2):
                base_split=base.split("_")
                base_before = base_split[0]+"_before_"+base_split[1]
                if not(self.generation in obj_stats):
                    obj_stats[self.generation]=0
                old_child1=self.previous_child_dir_path.split("/")[0] + "/"+str(int(self.previous_child_dir_path.split("/")[1])-1)
                self.faceMerge(old_child1+"/"+base, self.previous_child_dir_path+"/"+merge,
                                self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                self.faceMerge(old_child1+"_before/"+base_before, self.previous_child_dir_path+"_before/"+before,
                                self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_crossover.jpg",self.flip_images,self.contrast,self.feature_rate)
                #image = Image.open(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                imcv2=cv2.imread(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                val=predict_with_model(imcv2,self.model)
                pred=val[0]
                val1=val[1].data.cpu().numpy()[0]
                fake_confidence=val1[1]*100
                real_confidence=val1[0]*100
                family = {
                    'id': self.generated_count,
                    'Generation': self.generation,
                    'Type': 'Crossover',
                    'Current': str(self.generated_count) + "_crossover.jpg",
                    'Parent1': old_child1+"/"+base,
                    'Parent2': self.previous_child_dir_path+"/"+merge,
                    'Feature_Rate': self.feature_rate,
                    'Prediction':pred,
                    'Fake_Confidence':fake_confidence,
                    'Real_Confidence':real_confidence,
                    'Before':str(self.generated_count) + "_before_crossover.jpg"
                }


                if(self.generation in obj_total):
                    obj_total[self.generation]+=1
                else:
                    obj_total[self.generation]=1
                if(pred==0 and real_confidence>self.real_confidence_val):
                    if(self.generation in obj_acc):
                        obj_acc[self.generation]+=1
                    else:
                        obj_acc[self.generation]=1
                    objects_array.append(family)
                    self.generated_count += 1
                    self.num_generated_per_gen += 1
                    print("Crossover")
                else:
                    if(self.generation in obj_stats):
                        obj_stats[self.generation]+=1
                    else:
                        obj_stats[self.generation]=1
                    os.remove(self.next_child_dir_path+"/" + str(self.generated_count) + "_crossover.jpg")
                    os.remove(self.next_child_dir_path+"_before/" + str(self.generated_count) + "_before_crossover.jpg")



device = torch.device("cuda")
model = torch.load("full/xception/full_raw.p")
model.to(device)

for x in range(0,11):
    objects_array = []
    family_tree = {}
    obj_stats = {}
    obj_acc = {}
    obj_total = {}
    alpha = x/10 
    if not(os.path.exists("output_"+str(alpha))):
        os.mkdir("output_"+str(alpha))
    GAFM = GeneticAlgorithmFaceMerge("male_before","gan-male-backup", "male_after", "output_"+str(alpha),model,alpha, 5, 10, 0.05,True,0,85)
    GAFM.GeneticAlgorithmMerge()  





dir_array=["FINAL_RESULTS_GA/output_0.0",
"FINAL_RESULTS_GA/output_0.1",
"FINAL_RESULTS_GA/output_0.2",
"FINAL_RESULTS_GA/output_0.3",
"FINAL_RESULTS_GA/output_0.4",
"FINAL_RESULTS_GA/output_0.5",
"FINAL_RESULTS_GA/output_0.6",
"FINAL_RESULTS_GA/output_0.7",
"FINAL_RESULTS_GA/output_0.8",
"FINAL_RESULTS_GA/output_0.9",
"FINAL_RESULTS_GA/output_1.0"
]
with open('st.txt', 'w+') as ff:
    for k in dir_array:
        print(k)
        print("------------------------------------------")
        ff.write("---------------"+k+"-----------------\n")
        for x in os.listdir(k):
            print(k+"/"+x)
            if(os.path.isdir(k+"/"+x)):
                if("_" in x):
                    generation=x.split('_')[0]
                else:
                    generation=x
                ff.write("Generation: "+generation+"\n")
                f = open(k+"/"+x+"/st.json")
                data = json.load(f)   
                print(data)             
                ff.write("\t"+str(data)+"\n")
                ff.write("\t"+"{:.2f}".format((data['Found']/(data['count']))*100)+"\n")
                f.close()
        print("------------------------------------------")
        ff.write("------------------------------------\n") 



with open('ss.txt', 'w+') as ff:
    for k in dir_array:
        for x in os.listdir(k):
            if(x=="rejected.json"):
                f = open(k+'/rejected.json')
                data = json.load(f)
                print(data)                
                ff.write("-----------"+k+"-------------------\n")
                ff.write("{:.2f}".format((data['0']/(data['0']+300))*100)+"\n")
                ff.write("{:.2f}".format((data['1']/(data['1']+300))*100)+"\n")
                ff.write("{:.2f}".format((data['2']/(data['2']+300))*100)+"\n")
                ff.write("{:.2f}".format((data['3']/(data['3']+300))*100)+"\n")
                ff.write("{:.2f}".format((data['4']/(data['4']+300))*100)+"\n")
                ff.write("------------------------------------\n")
                f.close()





male_before=os.listdir("male_before")
male_after=os.listdir("male_after")
images_known = []
for x in male_before:
    images_known.append("male_before/"+x)

for x in male_after:
    images_known.append("male_after/"+x)
known_face_encodings = []
known_face_names = []
for x in images_known:
    known_image = face_recognition.load_image_file(x)
    known_face_encoding = face_recognition.face_encodings(known_image,model="large")[0]
    known_face_encodings.append(known_face_encoding)
    known_face_names.append(os.path.basename(x))


print('Learned encoding for', len(known_face_encodings), 'images.')

for root in dir_array:

    match_list={}
    for x in tqdm(os.listdir(root)):
        if(os.path.isdir(root+"/"+x)):
            count=0
            unk=0
            found=0
            for y in tqdm(os.listdir(root+"/"+x)):
                if(".jpg" in y):

                    unknown_image = face_recognition.load_image_file(root+"/"+x+"/"+y)


                    face_locations = face_recognition.face_locations(unknown_image,model="cnn")
                    face_encodings = face_recognition.face_encodings(unknown_image, face_locations,model="large")


                    pil_image = Image.fromarray(unknown_image)

                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                        name = "Unknown"

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        match_list[root+"/"+x+"/"+y]=name

                        if(name=="Unknown"):
                            unk+=1
                        else:
                            found+=1
                        count+=1


            st = {}

            st["Unknown"]=unk
            st["Found"]=found
            st["count"]=count

            with open(root+"/"+x+"/face_rec.json", "w") as write_file:
                json.dump(match_list, write_file, indent=4)

            with open(root+"/"+x+"/st.json", "w") as write_file:
                json.dump(st, write_file, indent=4)




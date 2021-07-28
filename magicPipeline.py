from deSkew import *
import cv2
import pytesseract
import spacy
import os




def imagePreTreatment(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15) #image en noir et blanc
    adaptive_threshold = cv2.bitwise_not(adaptive_threshold) #inverse les couleurs

    img = correctSize(adaptive_threshold) #verifie si la taille est suffisament grande pour extraire le texte
 
    return img


def getTextFromImage(img):
    custom_config = r'--oem 3 --psm 6' #oem3 =toutes les ressouces oem6=image en tant que texte (bloc texte)
    text = pytesseract.image_to_string(img, config=custom_config)

    text = text.replace('"', '`')
    text = text.replace("ï¿½", "?")
    text = text.replace("�", "?")
    text = text.replace('\\', '/')
    text = text.replace('\\', '/')

    return text


def getEntitiesFromText(text):

    
    nlp = spacy.load("../outputAccuracyT_V/model-last")
    

    line = text

    TOTAL = []

    total = ""
    date = ""
    company = ""
    doc = nlp(line)
    for i, ent in enumerate(doc.ents):
        print(i, "----", ent.text, ent.start_char, ent.end_char, ent.label_)
        if ent.label_ == "TOTAL":
            print("total found:", ent.text)
            total_here = ent.text
            if ',' in total_here:
                total_here = total_here.replace(',','.')
            try:    
                TOTAL.append(float(total_here))
            except Exception as e:
                print("Erreur sur float on string dans magicpipline, traitement ignoré car mot = " + total_here)
            
        if ent.label_ == "DATE":
            # print("DATE")
            date = ent.text
        if ent.label_ == "COMPANY":
            # print("COMPANY")
            company += ent.text

    if TOTAL:
        print("TOTAL : ---->", TOTAL)

        total = max(TOTAL)


    return company, date, total


def magicPipeline(image):
    return(getEntitiesFromText(getTextFromImage(imagePreTreatment(image))))








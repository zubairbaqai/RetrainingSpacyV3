import spacy
from spacy.training import Example
import json

class TrainSpacy():

    def __init__(self):
        try:
            self.nlp = spacy.load("New_web_lg")
        except:
            print("No Model Found , Using original model")
            self.nlp = spacy.load("en_core_web_lg")
        self.optimizer = self.nlp.resume_training()

    def TrainExample(self,Sentence,Entity=[(15, 20, "PERSON "), (25, 33, "ORG")]): ### Entity is a list in form of [(15, 20, "PERSON "), (25, 32, "PERSON")] in a sentence
        #Entity=[(15, 20, "PERSON "), (25, 32, "PERSON")]
        try:
            with open("Examples.txt", 'r') as f:
                TRAIN_DATA = json.load(f)
        except:
            TRAIN_DATA=[]


        Tuple=[]
        Tuple.append(Sentence)
        Tuple.append({"entities":Entity})
        Tuple=tuple(Tuple)

        ExistingExample=False
        for i in TRAIN_DATA:
            if(i[0]==Tuple[0]):
                ExistingExample=True
        if(not ExistingExample):
            TRAIN_DATA.append(Tuple)
        examples = []

        with open("Examples.txt", "w") as fp:  # Pickling
            json.dump(TRAIN_DATA, fp, indent=2)



        doc = self.nlp("Follow-up with Scott and Thishan on the call recording for the call centre")
        print(doc)
        for ent in doc.ents:
            print(ent.text, ent.label_)

        with self.nlp.select_pipes(enable="ner"):
            losses = {}

            for iteration in range(20):

                # shuufling examples  before every iteration
                for text, annots in TRAIN_DATA:
                    examples.append(Example.from_dict(self.nlp.make_doc(text), annots))

                    losses = self.nlp.update(examples, losses=losses, sgd=self.optimizer)
                    # print(losses)



        modelfile = 'New_web_lg'
        self.nlp.to_disk(modelfile)

    def TestExample(self,"Sentence "):
        doc = self.nlp("Follow-up with Scott and Thishan on the call recording for the call centre")
        print(doc)
        for ent in doc.ents:
            print(ent.text, ent.label_)


Train=TrainSpacy()
Train.TrainExample("Follow-up with Scott and Thishan on the call recording for the call centre")
Train.TestExample("Follow-up with Scott and Thishan on the call recording for the call centre")







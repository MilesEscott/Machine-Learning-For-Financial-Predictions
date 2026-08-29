import pandas
import random

class Node:
    
    def __init__(self):
        self.__left = None
        self.__right = None
        self.__question = None

    def getQuestion(self):
        return self.__question
    
    def getLeft(self):
        return self.__left
    
    def getRight(self):
        return self.__right
    
    def setQuestion(self, question):
        self.__question = question

    def setLeft(self, left):
        self.__left = left

    def setRight(self, right):
        self.__right = right

class Leaf:

    def __init__(self,values):
        self.__valuesOfEachLabel = values
        
        self.__prediction = ""
        count = 0
        for each in values:
            if values[each] > count:
                self.__prediction = each
                count = values[each]
    
    def getPrediction(self):
        return self.__prediction
    
    def getvaluesOfEachLabel(self):
        return self.__valuesOfEachLabel

class Question:

    def __init__(self,column,value):
        self.__column = column
        self.__value = value

    def match(self, data):
        val = data[self.__column]
        return val >= self.__value
    
    def getColumn(self):
        return self.__column
    
    def getValue(self):
        return self.__value
    
    def __str__(self):
        return f"Is {self.__column} >= {self.__value}"

def PrintTree(node, spacing=""):
    
    if isinstance(node, Leaf):
        print(spacing + "Prediction: ", node.getPrediction())
        return
    
    # If no question, it's a leaf
    if node.getQuestion() is None:
        print(spacing + "Leaf")
        return
    
    # Print current question
    print(spacing + str(node.getQuestion()))
    
    # Left branch (True)
    print(spacing + "├── True:")
    PrintTree(node.getLeft(), spacing + "│   ")
    
    # Right branch (False)
    print(spacing + "└── False:")
    PrintTree(node.getRight(), spacing + "    ")

def CountClasses(dataset, label):
    counts = {}
    for entry in range(0, len(dataset)):
        if dataset.iloc[entry][label] not in counts:
            counts[dataset.iloc[entry][label]] = 0
        counts[dataset.iloc[entry][label]] += 1
    
    return counts

def Gini(dataset, label, rows = None):
    
    impurity = 0
    
    if rows != None:
        dataset = dataset.iloc[rows]

    allTypesOfLabels = []
    for entry in range(0,len(dataset)):
        if not dataset.iloc[entry][label] in allTypesOfLabels:
            allTypesOfLabels.append(dataset.iloc[entry][label])


    for typeOfLabel in allTypesOfLabels:
        noOfType = len(dataset[dataset[label] == typeOfLabel])
        pi = noOfType/len(dataset)
        impurity += (pi**2)

    impurity = 1 - impurity

    return impurity 

def ComputeInfoGained(dataset,question, label):
    
    # Take left to be >= value
    left = []
    right = []
    
    for index in range(0, len(dataset)):
        if question.match(dataset.iloc[index]):
            left.append(index)
        else:
            right.append(index)
    
    if left == [] or right == []:
        return left,right,0
    
    infoGained = Gini(dataset, label) - ((len(left)/len(dataset))* Gini(dataset, label, left) + (len(right)/len(dataset))* Gini(dataset, label, right))

    return left, right, infoGained

def CreateSetsOfQuestions(dataset, label):
    questions = {}
    for col in dataset.columns:
        if col != label:
            thresholds = dataset[col].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
            tempQuestions = []
            for entry in thresholds:
                tempQuestions.append(Question(col, entry))
            questions[col] = tempQuestions
    
    return questions

def FindBestQuestion(dataset, possibleQuestions, label):
    
    mostInfo = 0
    bestQuestion = None
    finalLeft = []
    finalRight = []

    for columnsOfQuestions in possibleQuestions.values():
        for question in columnsOfQuestions:
            left, right, infoGained = ComputeInfoGained(dataset,question, label)
            if infoGained > mostInfo:
                bestQuestion = question
                mostInfo = infoGained
                finalLeft = left
                finalRight = right
    
    return bestQuestion, finalLeft, finalRight

def TreeBuilder(dataset,maxDepth, allQuestions, label):

    currentNode = Node()
    
    if Gini(dataset, label) == 0:
        return Leaf(CountClasses(dataset, label))
    
    foundBestQuestion = None
    leftDatasetIndexs = []
    rightDatasetIndexs = []
        
    foundBestQuestion,leftDatasetIndexs,rightDatasetIndexs = FindBestQuestion(dataset, allQuestions, label)
    if foundBestQuestion == None:
        return currentNode
    currentNode.setQuestion(foundBestQuestion)


    if maxDepth > 0:

        # Left Direction
        
        currentNode.setLeft(TreeBuilder(dataset.iloc[leftDatasetIndexs],maxDepth-1, allQuestions, label))
        
        # Right Direction
        currentNode.setRight(TreeBuilder(dataset.iloc[rightDatasetIndexs],maxDepth-1, allQuestions, label))
    
    else:
        currentNode = Leaf(CountClasses(dataset,label))


    return currentNode

def Classifying(root,datapoint,label):
    currentNode = root
    while not isinstance(currentNode, Leaf):
        
        currentQuestion = currentNode.getQuestion()
        
        if currentQuestion.match(datapoint):
            currentNode = currentNode.getLeft()
        else:
            currentNode = currentNode.getRight()
    
    if currentNode.getPrediction() == datapoint[label]:
        return 1
    else:
        return 0
    
def Testing(root, dataset, label):
    noOfTests = len(dataset)
    correct = 0
    predictions = []
    for entry in range(0,noOfTests):

        result = Classifying(root, dataset.iloc[entry], label)
        predictions.append(result)
        correct += result

    dataset["Prediction"] = predictions

    print()
    print("Performing a test with", noOfTests, "datapoints")
    print("The model got", correct, "datapoints correct")
    print("Achieving an accuracy of", correct*100/noOfTests,"%")
    return dataset, correct

def BaggingSample(dataset):
    col1 = random.randint(0,len(dataset.columns)-2)
    col2 = -1
    
    while col2 == -1 or col2 == col1:
        col2 = random.randint(0,len(dataset.columns)-2)
    
    datapoints = []
    for i in range(0,len(dataset)):
        datapoints.append(random.randint(0,len(dataset)-1))

    newSample = dataset.iloc[datapoints,[col1,col2,-1]]

    return newSample

def RandomForestBuilder(dataset,label, noOfTrees, maxDepth):
    
    roots = []
    for i in range(0,noOfTrees):
        
        sample = BaggingSample(dataset)
        allQuestions = CreateSetsOfQuestions(sample, label)
        roots.append(TreeBuilder(sample,maxDepth,allQuestions, label))

    return roots

def MajorityVoteCounter(roots,datapoint,label):

    counts = {}
    
    for root in roots:
        currentNode = root
        while not isinstance(currentNode, Leaf):
            
            currentQuestion = currentNode.getQuestion()
            
            if currentQuestion.match(datapoint):
                currentNode = currentNode.getLeft()
            else:
                currentNode = currentNode.getRight()
    
        if not currentNode.getPrediction() in counts:
            counts[currentNode.getPrediction()] = 0
        counts[currentNode.getPrediction()] += 1

    highestCount = 0
    finalPrediction = None
    for prediction in counts:
        if counts[prediction] > highestCount:
            finalPrediction = prediction
            highestCount = counts[prediction]

    if finalPrediction == datapoint[label]:
        return 1
    else:
        return 0
    
def FinalTest(roots,dataset,label):
    
    noOfTests = len(dataset)
    correct = 0
    predictions = []
    for entry in range(0,noOfTests):

        result = MajorityVoteCounter(roots, dataset.iloc[entry], label)
        predictions.append(result)
        correct += result
    
    dataset["Prediction"] = predictions

    print()
    print("Performing a test with", noOfTests, "datapoints")
    print("The model got", correct, "datapoints correct")
    print("Achieving an accuracy of", correct*100/noOfTests,"%")
    return dataset, correct

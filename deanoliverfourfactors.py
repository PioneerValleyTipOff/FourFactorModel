#expect to make changes to this script as needed. need to modify to add future data
#along with historical data. if possible, try to acquire other high school teams
#data or ask the teams for data from previous season for guidance.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#make sure that xlsxwriter, openpyxl, and sklearn are installed
import pandas as pd #read in the dataset and use certain columns
import numpy as np #added when working with datasets like this one to complete operations faster
from sklearn.model_selection import train_test_split #needed to train and test model
from sklearn.linear_model import LogisticRegression #model of choice
from sklearn.metrics import confusion_matrix #assess if easier to predict loss or win more accurately
from sklearn.metrics import ConfusionMatrixDisplay #visual for the matrix
from matplotlib import pyplot as plt #visualization tool to use
import seaborn as sns #another visual tool
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#only using the Dean Oliver 4 Factors as we want to analyze the impact that this
#can possibly have on the outcome of these games, most specifically to PVTO
four_factors = ['Efg%', 'Opp_Efg%',	'Ft/Fga',	'Opp_Ft/Fga', 'RebO%',	'RebD%',	'To%', 'Opp_To%']

#add the chosen path of desire
def prepare_data(path):
  dataset = pd.read_excel(path+'/PVTOData.xlsx', sheet_name='TeamData')

  #using the amount of points to determine win or loss from each game
  team_points = dataset['Points']

  result = []

  #using 0 for loss and 1 for win
  for i in range(0,len(team_points),2):
    win = 1
    loss = 0
    #add the result of the game to the list. make sure that there are no ties
    #since that doesnt exist in basketball. 
    if team_points[i] > team_points[i+1]:
      result.append(win)
      result.append(loss)
    else:
      result.append(loss)
      result.append(win)
  
  #have dataset with team name and without team name for further analysis
  factors = dataset[four_factors]
  factors.insert(len(factors.columns), 'Result', result)

  return [factors, dataset['Name']]

#copy-paste your desired file path here
file_path = ' '

#keeping the team name for the last function in this script
model_data, team_name = prepare_data(file_path)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Since win/loss is a binary variable, the model of choice is a logisitc 
#regression model. We want to use the logistic regression to see how accurate it will be over 
#many different iterations. This function will consist of running model and gathering
#different meausrements to use
def logit_model(N): 
  #model accuracy
  scores = [] 

  #confusion matrix
  error_behavior = np.array([[0,0],
                             [0,0]])
  #model coefficients
  intercept = []
  coeff = []
  
  i = 0
  while i < N:
    #since we are thinking probabilistic, we want to be able to generate random data all of the time
    #the test_size is random generated between .25 and .45, which means that we will have anywhere 
    #between 10-14 teams as training data and anywhere between 4-8 teams as training data

    #once this dataset is large enough, using a set test and training data size will be much more useful
    X_train, X_test, y_train, y_test = train_test_split(model_data[four_factors], 
                                                        model_data['Result'], 
                                                        test_size=np.random.uniform(.25,.45),
                                                        random_state = i)
    logit = LogisticRegression().fit(X_train,y_train)

    scores.append(logit.score(X_test,y_test))

    error_behavior += confusion_matrix(y_test, logit.predict(X_test), labels=logit.classes_)

    intercept.append(logit.intercept_)
    coeff.append(logit.coef_)
    
    i+=1

  #after running the model, we get the averages which is used in later function
  error_behavior = np.round(error_behavior/N,4)
  
  coefficients = pd.concat([pd.DataFrame(np.concatenate(intercept), columns=['Intercept']),
                          pd.DataFrame(np.concatenate(coeff), columns=four_factors)],axis=1)
  
  parameters = list(coefficients.columns)
  
  #taking the average of the coefficients from all of the runs of the model
  model_coefficients = [np.round(coefficients[j].mean(),6) for j in parameters]

  #vital for further anaylsis
  return [scores, error_behavior, model_coefficients, parameters]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
analyze_model = logit_model(10000)

#frequency of accuracy bundled into 10 groups. helps paint the picture what is happening
#after the model 10,000 times from above
def model_score(scores):
  sns.histplot(scores,bins=10)
  plt.xlim(0,1)
  plt.xlabel('Accuracy')
  plt.grid(False)
  plt.rcParams["figure.figsize"] = (10,8)
  plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#display the average win-loss guesses and use this to calculate percent accuracy 
#for a team to lose or win a game with the four factors. use the visualization to 
#help interpret what is happening within the model
def prediction_matrix(grid):
  dis = ConfusionMatrixDisplay(confusion_matrix=grid, 
                               display_labels=['Lose', 'Win'])
  dis.plot(values_format='.4f')
  plt.grid(False)
  plt.rcParams["figure.figsize"] = (10,8)
  plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#print out the equation for the paper
def display_equation(coeff,factors):
  equation = ' w = ' + str(coeff[0]) + ' + '

  for i in range(1,len(factors)):
    equation += '(' + str(coeff[i]) + '*' + factors[i] + ')'
    if i < len(factors)-1:
      equation += ' + '

  return equation
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#displays which features are most important for positive and negative values
def feature_importance(coeff,factors):
  #adding the model coefficent values to the graph. this is to visualize the 
  #feature importance (most important to team success)
  fig, ax = plt.subplots()

  #color is blue if positive and red when negative
  pps = ax.bar(factors,coeff,color=['blue' if j >= 0 else 'red' for j in coeff])
  for p in pps:
   ax.annotate('{}'.format(p.get_height()),xy=(p.get_x() + p.get_width() / 2, p.get_height()),
               xytext=(0, 3),textcoords="offset points", ha='center', va='bottom')

  plt.ylim(-.5,.5)
  plt.ylabel('Model Coefficient Values')
  plt.rcParams["figure.figsize"] = (10,8)
  plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#devise ranking system using the model  
def rank_system(path,coeff,factors):
  weights = dict(zip(factors, np.round(coeff,2)))

  #in this ranking system, we are seeing how the teams perform in which there is 
  #no dependency on how the opposing team played over the course of the game. this
  #can show that a team can play well in a loss or poorly in a win.
  for i in factors:
    if 'Opp' in i:
      del weights[i]
      model_data.drop([i], axis=1, inplace=True)

  model_data.drop(['Result'], axis=1, inplace=True)

  total_weights = np.round(sum(weights.values()),2)

  if total_weights > 1:
    diff = total_weights - 1
  else:
    diff = 0

  negate_lowest_value = -1*weights['To%']
  efg_three_factor = np.round(weights['Efg%']*.5,2)
  max_value = total_weights-diff+negate_lowest_value+efg_three_factor

  #using the formula from the paper without opposing team stats
  score = model_data.dot(pd.Series(weights)) + negate_lowest_value 

  #to get a more decifering score, we multiply it by 10. as referenced in the paper,
  #we are trying to create a ranking system from 0-10 in which 10 is absolute 
  #perfection while 0 is where the team did absolutely everything wrong
  rank = np.round((10*(score/max_value)),1)

  ratings = list(rank)
  for i in range(len(ratings)):
    if ratings[i] > 10:
      ratings[i] = 10*(ratings[i]*(1/ratings[i]))
    else:
      continue

  teams_score = pd.concat([team_name,pd.DataFrame(ratings)],axis=1)
  teams_score.columns = ['Team', 'Score']

  #separate the rankings into girls and boys teams so that we are ranking the teams
  #in relative to what their competition level would be like
  girls_data = teams_score[teams_score['Team'].str.endswith('G')]
  boys_data = teams_score[teams_score['Team'].str.endswith('B')]

  writer = pd.ExcelWriter(path + '/RankData.xlsx', engine='xlsxwriter')
  writer.save()

  #puts each of the player and team data into separate sheets
  with pd.ExcelWriter(path + '/RankData.xlsx', engine='openpyxl', mode='a') as writer:
        girls_data.sort_values(by='Score', ascending=False).to_excel(
            writer, sheet_name='Girls', index=False)
        boys_data.sort_values(by='Score', ascending=False).to_excel(
            writer, sheet_name='Boys', index=False)
  
  return 'Rank data uploaded.'
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions to run. to keep it safe, run them in order as this isn't perfect
#designed script. improvements expected to be made upon what has been done.
model_coeff = analyze_model[2]
model_parameters = analyze_model[3]

model_score(analyze_model[0])
prediction_matrix(analyze_model[1])
display_equation(model_coeff, model_parameters)

#removing model intercept as it's not used when ranking the teams or seeing which 
#features were the most important for the model
model_parameters.remove(model_parameters[0])
model_coeff.remove(model_coeff[0])

feature_importance(model_coeff, model_parameters)
rank_system(file_path,model_coeff,model_parameters)

<a id='section_6'></a>
<img src=" image url here ">


<h1><center>NBA Classification Project</center></h1>
<center> Author: Gilbert Noriega </center>

[About the Project](#section_1) [Data Dictionary](#section_2) [Initial Hypotheses/Thoughts](#section_3) [Project Plan](#section_4) [How to Reproduce](#section_5)



<a id='section_1'></a>
## About the Project
___

### Background
> Add some background

___
>*Acknowledgement:The dataset was provided by Codeup from the MySequel Database* 

___

### Goals
> My goal for this project is to create a model that will accuractely predict wins and losses for NBA teams during the 2014 and 2018 seasons. I will deliver the following in a github repository: 
>
> - A clearly named final notebook. This notebook will be what I present and will contain plenty of markdown documentation and cleaned up code.
> - A README that explains what the project is, how to reproduce you work, and your notes from project planning
> - A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.
  
[back to the top](#section_6)

___

<a id='section_2'></a>
## Data Dictionary

| Features | Definition |
| :------- | :-------|
| |  |
| |  |

|  Target  | Definition |
|:-------- |:---------- |
|  W  |  wins column for the team |

[back to the top](#section_6)
___
<a id='section_3'></a>
## Initial Hypothesis & Thoughts

>### Thoughts
>
> - We could add a new feature?
> - Should I turn the continuous variables into booleans?

>### Hypothesis
> - Hypothesis 1:
>   - H<sub>0</sub>: 
>   - H<sub>a</sub>: 
>
> - Hypothesis 2:
>   - H<sub>0</sub>: 
>   - H<sub>a</sub>: 
>
> - Hypothesis 3:
>   - H<sub>0</sub>: 
>   - H<sub>a</sub>: 
>
> 

[back to the top](#section_6)
___
<a id='section_4'></a>
## Project Plan: Breaking it Down

>- acquire
>    - acquire data from csv
>    - turn into a pandas dataframe
>    - summarize the data
>    - plot distribution
>
>- prepare
>    - address data that could mislead models
>    - create features
>    - split into train, validate, test
>
>- explore
>    - test each hypothesis
>    - plot the continuous variables
>    - plot correlation matrix of all variables
>    - create clusters and document its usefulness/helpfulness
> 
>- model and evaluation
>    - which features are most influential: use rfe
>    - try different algorithms: LinearRegression, Decision Tree, Random Forest, K-Nearest Neighbor
>    - evaluate on train
>    - evaluate on validate
>    - select best model
>    - create a model.py that pulls all the parts together.
>    - run model on test to verify.
>
>- conclusion
>    - summarize findings
>    - make recommendations
>    - next steps


[back to the top](#section_6)

___

<a id='section_5'></a>
## How to Reproduce

>1. Download data from here <insert link>
>2. Install prepare.py and model.py into your working directory.
>3. Run a jupyter notebook importing the necessary libraries and functions.
>4. Follow along in final_report.ipynb or forge your own exploratory path. 

[back to the top](#section_6)

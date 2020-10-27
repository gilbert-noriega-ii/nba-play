import pandas as pd
import numpy as np 
import os

from sklearn.model_selection import train_test_split




def prep_nba(season):
    '''
    This function creates new columns and deletes unneeded columns
    '''
    #splits teams and opponents into respective conferences
    nba = conference_split(season)

    #changing conference into dummy variable
    conference = pd.get_dummies(nba.Conference, drop_first = True)

    #changing Opp.Conference into dummy variable
    oppconference = pd.get_dummies(nba['Opp.Conference'], drop_first = True)
    
    #changing home into dummy variable
    home = pd.get_dummies(nba.Home, drop_first = True)

    #changing target variable into dummy variable
    wins = pd.get_dummies(nba.WINorLOSS, drop_first = True)

    #dropping unnecessary columns
    nba = nba.drop(columns = ['Unnamed: 0', 'Home', 'WINorLOSS', 'Date', 'Game', 'TeamPoints', 'OpponentPoints', 'FieldGoals', 'Opp.FieldGoals', 'FieldGoalsAttempted', 'Opp.FieldGoalsAttempted', 'X3PointShots', 'Opp.3PointShots', 'X3PointShotsAttempted', 'Opp.3PointShotsAttempted', 'FreeThrows', 'Opp.FreeThrows', 'FreeThrowsAttempted', 'Opp.FreeThrowsAttempted', 'Conference', 'Opp.Conference'])

    #adding dummy variables back into the main dataframe
    nba = pd.concat([nba, conference, oppconference, home, wins], axis = 1)

    return nba


def conference_split(nba):
    '''
    This functioin splits the Team and Opponent into Conferences.
    '''
    #setting conditions for conference
    conditions = [
        #west teams
        (nba.Team.isin(['LAL', 'LAC', 'DEN', 'HOU', 'OKC', 'UTA', 'DAL', 'POR', 'MEM', 'PHO', 'SAS', 'SAC', 'NOP', 'MIN', 'GSW'])),
        #east teams
        (nba.Team.isin(['MIL', 'TOR', 'BOS', 'IND', 'MIA', 'PHI', 'BRK', 'ORL', 'WAS', 'CHO', 'CHI', 'NYK', 'DET', 'ATL', 'CLE']))]
    choices = ['home_is_west', 'home_is_east']
    #creating conference column for home team
    nba['Conference'] = np.select(conditions, choices, default='west')
    #setting conditions for oppConference
    conditions2 = [
        #west teams
        (nba.Opponent.isin(['LAL', 'LAC', 'DEN', 'HOU', 'OKC', 'UTA', 'DAL', 'POR', 'MEM', 'PHO', 'SAS', 'SAC', 'NOP', 'MIN', 'GSW'])),
        #east teams
        (nba.Opponent.isin(['MIL', 'TOR', 'BOS', 'IND', 'MIA', 'PHI', 'BRK', 'ORL', 'WAS', 'CHO', 'CHI', 'NYK', 'DET', 'ATL', 'CLE']))]
    choices2 = ['away_is_west', 'away_is_east']
    #creating oppConference column for away team
    nba['Opp.Conference'] = np.select(conditions2, choices2, default='west')
    return nba


def season_split(nba):
    '''
    This function splits the data into different nba seasons
    '''
    #nba 14-15 season
    nba14_15 = nba.loc[:2459,:]
    #nba 15-16 season
    nba15_16 = nba.loc[2460:4919,:]
    #nba 16-17 season
    nba16_17 = nba.loc[4920:7379,:]
    #nba 17-18 season
    nba17_18 = nba.loc[7380:,:]
    
    return nba14_15, nba15_16, nba16_17, nba17_18

def nba_split(df):
    '''
    This function splits a dataframe into train, validate, and test sets
    '''
    train_and_validate, test = train_test_split(df, train_size=.8, random_state=123, stratify=df.W)
    train, validate = train_test_split(train_and_validate, train_size = .7, random_state=123, stratify=train_and_validate.W)
    return train, validate, test

def wrangle_nba_14_15():
    '''
    This function splits the nba14_15 season into train, validate, and test sets
    '''
    #save csv as a dataframe
    nba = pd.read_csv('nba.games.stats.csv')
    #preppring dataframe nba stats
    nba = prep_nba(nba)
    #splitting into seasons
    nba14_15, nba15_16, nba16_17, nba17_18 = season_split(nba)
    #splitting into train, validate test
    train, validate, test = nba_split(nba14_15)
    return train, validate, test

def wrangle_nba_15_16():
    '''
    This function splits the nba15_16 season into train, validate, and test sets
    '''
    #save csv as a dataframe
    nba = pd.read_csv('nba.games.stats.csv')
    #preppring dataframe nba stats
    nba = prep_nba(nba)
    #splitting into seasons
    nba14_15, nba15_16, nba16_17, nba17_18 = season_split(nba)
    #splitting into train, validate test
    train, validate, test = nba_split(nba15_16)
    return train, validate, test

def wrangle_nba_16_17():
    '''
    This function splits the nba16_17 season into train, validate, and test sets
    '''
    #save csv as a dataframe
    nba = pd.read_csv('nba.games.stats.csv')
    #preppring dataframe nba stats
    nba = prep_nba(nba)
    #splitting into seasons
    nba14_15, nba15_16, nba16_17, nba17_18 = season_split(nba)
    #splitting into train, validate test
    train, validate, test = nba_split(nba16_17)
    return train, validate, test

def wrangle_nba_17_18():
    '''
    This function splits the nba17_18 season into train, validate, and test sets
    '''
    #save csv as a dataframe
    nba = pd.read_csv('nba.games.stats.csv')
    #preppring dataframe nba stats
    nba = prep_nba(nba)
    #splitting into seasons
    nba14_15, nba15_16, nba16_17, nba17_18 = season_split(nba)
    #splitting into train, validate test
    train, validate, test = nba_split(nba17_18)
    return train, validate, test
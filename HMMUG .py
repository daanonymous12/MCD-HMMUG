#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 01:02:21 2019

@author: da
"""
import numpy as np
import scipy.stats as stats
import random 
from math import log as ln

class hidden_markov:

    def __init__(self, data):
        self.data = np.asarray(data).astype(np.float)

    def xMarkovChainEquilibrium(self,mnTransitionMatrix):
        # initialize the transitional matrix 
        temp = np.transpose(mnTransitionMatrix) - np.identity(len(mnTransitionMatrix)) + 1
        return np.sum(np.linalg.inv(temp),axis = 1)

    def start(self,istates,LikelihoodTolerance,MaxIteration):
        # inittalize random matrix to calculate mnA
        random_matrix = np.empty((istates, istates))
        for i in range(istates):
            for j in range(istates):
                random_matrix[i][j] = random.uniform(.01,.99)
        mnA = random_matrix / np.sum(random_matrix,axis = 1)[:,np.newaxis]
        # compute parameters iota
        vnIota = self.xMarkovChainEquilibrium(mnA)
        # finding mean and stdev for each state
        random_list = np.empty((1,istates))
        for i in range(istates):
            random_list[0][i] = random.uniform(-.99,.99)
        vnM = (1/ np.sum(random_list)) * random_list * np.mean(self.data)
        for i in range(istates):
            random_list[0][i] = random.uniform(.1,.99)
        vnS = (1/np.sum(random_list) * random_list * np.var(self.data,axis = 0))**.5
        # start hidden markov with variables initilized 
        self.xHMMUG(vnIota,mnA,vnM,vnS,LikelihoodTolerance,MaxIteration)
        
    def parameter_calculation(self,iN,iT,vnMean,vnSdev,vnIota,mnA):
        mnB = np.empty((iT,iN))
        for k in range(iT):
            for i in range(iN):
                mnB[k][i] = stats.norm(vnMean[0][i],vnSdev[0][i]).pdf(self.data[k])
        # initialize zero array and initial estimates for alpha and Nu 
        mnAlpha = np.zeros((iT,iN))
        vnNu = np.zeros((1,iT))
        mnAlpha[0] = vnIota*mnB[0]
        vnNu[0][0] = 1/np.sum(mnAlpha[0])
        mnAlpha[0] *= vnNu[0][0]
        # fill in the matrices
        for i in range(iT-1):
            mnAlpha[i+1] = np.dot(mnAlpha[i],mnA) * mnB[i+1]
            vnNu[0][i+1] = 1/np.sum(mnAlpha[i+1])
            mnAlpha[i+1] *= vnNu[0][i+1]
        # return mnAlpa,vnNu,mnB
        return mnAlpha,vnNu,mnB

    def xHMMUG(self,Iota,alpha,mean,sdev,LiklihoodTolerance,MaxIteration):
        nTol,iMaxIter = LiklihoodTolerance,MaxIteration
        # initialize variables such as alpha, length of data and others
        iT,iN,vnIota,mnA,vnMean,vnSdev = len(self.data),len(alpha),Iota,alpha,mean,sdev
        # initial table for probabilities based on normal distribution of given mean and stdev
        mnAlpha,vnNu,mnB = self.parameter_calculation(iN,iT,vnMean,vnSdev,vnIota,mnA)
        # log likelihood calculations 
        vnLogLikelihood = []
        vnLogLikelihood.append(-np.sum(np.log(vnNu)))
        for i in range(iMaxIter):
            # expectation step of the EM algorithm 
            # estimation of beta 
            mnBeta = np.zeros((iT,iN))
            mnBeta[iT-1] = vnNu[0][iT-1]
            counter = iT-2
            while counter >= 0:
                mnBeta[counter] = np.dot(mnA,mnBeta[counter+1]*mnB[counter+1])*vnNu[0][counter]
                counter -= 1
            # estimation of gamma
            mnGamma = (mnAlpha*mnBeta)/np.sum((mnAlpha*mnBeta),axis = 1)[:,np.newaxis]
            # estimation of xi
            mnXi = np.zeros((iN,iN))
            for i in range(iT-2):
                temp_xi = mnA * np.kron(mnAlpha[i],mnBeta[i+1]*mnB[i+1]).reshape((iN,iN))
                mnXi += (temp_xi/np.sum(temp_xi,axis = 1)[:,np.newaxis])
            # m step of algorithm 
            mnA = mnXi/np.sum(mnXi,axis = 1)[:,np.newaxis]
            # iota
            vnIota = mnGamma[0]
            # finding weights using gamma
            mnWeights = np.transpose(mnGamma)/np.sum(np.transpose(mnGamma),axis = 1)[:,np.newaxis]
            # mean and standard deviation
            vnMean = np.dot(mnWeights,self.data)
            vnMean = [vnMean]
            vnS_temp = []
            for i in range(len(vnMean[0])):
                vnS_temp.append((self.data - vnMean[0][i])**2)
            vnS_temp = np.array(vnS_temp)
            vnSdev = np.array([np.sum(mnWeights*(vnS_temp),axis = 1)])**.5
            # creation of gaussian normal with selected pdf, mean and stdev
            # repeating steps done before 
            mnAlpha,vnNu,mnB = self.parameter_calculation(iN,iT,vnMean,vnSdev,vnIota,mnA)
            vnLogLikelihood.append(-np.sum(np.log(vnNu)))
            # likelihood test for early breaking of loop
            if (vnLogLikelihood[-1]/vnLogLikelihood[-2])-1 <= nTol:
                break
        #BIC 
        nBIC = -2 * vnLogLikelihood[-1] + (iN * (iN+2)-1) * ln(iT)
        print('Iota = ', vnIota, '\n','a = ', mnA, '\n', 'mean = ', vnMean, '\n', 'stdev = ', vnSdev)
        print('gamma = ', mnGamma, '\n','BIC = ', nBIC, '\n', 'Log likelihood = ', vnLogLikelihood[-1])
        print('\n')
import numbers 
import numpy as np 
from scipy import linalg
import math
from random import choices
from scipy import stats

class fast_MCD:

    def __init__(self,data):
        self.data = np.array(data)
        self.ndatalength,self.numofcomponent = self.data.shape
        self.npointspartition = math.floor((self.ndatalength + self.numofcomponent + 1) / 2)

    def rescale_param(self,data_size):
        return data_size / stats.chi2.cdf(x = stats.chi2.ppf(data_size, df = self.numofcomponent), df = self.numofcomponent+2)

    def cstep(self,data,numcomponents,initialestimate):
        #dot product of 3 matrix to calculate distance
        last_distance = []
        #calculate distance of each row of data, append it onto an empty list 
        for i in range(len(data)):
            last_distance.append(np.dot(np.dot(data[i] - initialestimate[0], linalg.inv(initialestimate[1])),(data[i] - initialestimate[0])).item())
        ordered_dist = np.argsort(last_distance)
        index = []
        #generate new set of data to find new mean and cov estimates 
        #through smallest numofcomponents of rows of original data 
        for i in range(numcomponents):
            index.append(ordered_dist[i])
        new_data = data[[index]]
        new_mean = np.mean(new_data,axis = 0)
        cov_new = np.cov(new_data,rowvar = False) 
        #return mean, cov and of new data and row index of data used 
        return [new_mean,cov_new,index]

    def equalpartition(self):
            location = np.mean(self.data, axis = 0)
            spread = np.cov(self.data, rowvar = False)
            print('Location= ' + location + 'Spread= ' + spread)

    def onecomp(self,sortedx):
        mnUniSubSets = []
        for i in range(self.ndatalength - self.npointspartition+1):
            placeholder = []
            for j in range(i, i + self.npointspartition - 1):
                placeholder.append(sortedx[j,0])
                mnUniSubSets.append(placeholder)
        vnUnimeans = np.mean(mnUniSubSets)
        vnUniVar = np.var(mnUniSubSets)
        print('Location=' + vnUnimeans + ' Spread= ' + vnUniVar)

    def smalldata(self):
        #generate a table of 500 mean and cov estimates through OneCsteps
        CandidateEstimates = []
        Order = []
        for _ in range(500):

            #initial estimate throug the estimates generated from prev step
            #first estimate will use initial estimate while second estimate will use first 
            mxInitialEstimate = self.estimate(self.numofcomponent)
            mxFirstEstimate = self.cstep(self.data, self.npointspartition,mxInitialEstimate)
            mxSecondEstimate = self.cstep(self.data, self.npointspartition,[mxFirstEstimate[0],mxFirstEstimate[1]])
            #estimates will be put into a table and det of each cov 
            #estimate will be put into a list which will be ordered 
            CandidateEstimates.append(mxSecondEstimate)
            Order.append(np.linalg.det(mxSecondEstimate[1]))
        self.c_step_conv(CandidateEstimates,Order)

    def estimate(self,nIntSubSetSize):
        #generating initial estimates(mean and cov) through randomized index                 
        #obtained in previous step 
        #randomly generate # rows of indexes
        vnIntSubSet = choices(range(self.ndatalength), weights = None, cum_weights = None, k = self.ndatalength)
        while True: 
            empty_matrix = []
            nIntSubSetSize += 1
            for i in range(nIntSubSetSize):
                vnCurrentSubSet = self.data[vnIntSubSet[i]]
                empty_matrix.append(vnCurrentSubSet)
            empty_matrix = np.asarray(empty_matrix)
            tempcov = np.cov(empty_matrix,rowvar = False)
            if linalg.det(tempcov) > 0:
                return [np.mean(empty_matrix, axis = 0), tempcov]

    def c_step_conv(self, mxCandidateEstimates,mxorder):
        #sort the list of determinants and find the 10 set with lowest determinant of covariance
        mxCandidateEstimates = np.asarray(mxCandidateEstimates)
        mxorder = np.argsort(mxorder)
        tenordering = []
        mxorder = mxorder.tolist()
        for i in range(10):
            tenordering.append(mxorder.index(i))
        candidate = mxCandidateEstimates[tenordering]
        #carry out the c step for 10 results with lowest determinant
        mxFinalSet = []
        for i in candidate:
            mxCurrentEstimate = i
        #c step convergence for all 10 value, return value with lowest determinant
            while True:
                mxLastEstimate = mxCurrentEstimate
                mxCurrentEstimate = self.cstep(self.data,self.npointspartition,[mxLastEstimate[0],mxLastEstimate[1]] )
                if linalg.det(mxCurrentEstimate[1]) == 0 or (linalg.det(mxCurrentEstimate[1]) == linalg.det(mxLastEstimate[1])):
                    mxCurrentEstimate = mxLastEstimate
                    break
            mxFinalSet.append(mxCurrentEstimate)
        mxFinalSet = np.asarray(mxFinalSet)
        self.result(mxFinalSet)

    def result(self, mxFinalSet):
        #sort the convergent 10 estimates by determinant of covariance 
        determinant_calc = []
        rescale_param = self.rescale_param(self.npointspartition / self.ndatalength)
        for i in range(10):
            determinant_calc.append(linalg.det(mxFinalSet[i][1]))
        determinant_calc = np.argsort(determinant_calc)
        #obtain set of convergent estimates with smallest determinant 
        #and returning the ans to be outputed to user 
        mxFinalSet = mxFinalSet[determinant_calc[0]]
        fmean = mxFinalSet[0] 
        fcov = mxFinalSet[1]
        findex = mxFinalSet[2]
        scaledcov = rescale_param*fcov
        #output final result 
        print('Running MCD...Number of component=', self.numofcomponent,'Data Length=',self.ndatalength,'and Points Ellipse Covers=',self.npointspartition)
        print('Mean=>','\n', fmean)
        print('cov=>','\n', fcov)
        print('index=>','\n', findex)
        print('Scaled Cov =>','\n',scaledcov)

    def bigdata(self):
        nApproxSubSetLength=math.ceil(self.ndatalength/5)
        nDistjoinSet=[]
        for i in range(1):
            nDistjoinSet.append(self.data[i*nApproxSubSetLength:(i+1)*nApproxSubSetLength])
        nNumDisjointrows=len(nDistjoinSet)
        vnDisjointColumnIndex=list(map(len,nDistjoinSet))
        #repeating each subset calculation nNumDisjointrows times
        #generate a table of 500/nNumDisjointrows mean and cov through OneCsteps 
        mxEstimatesInsideSubsets=[]
        for z in range(nNumDisjointrows):
            mxCandidateEstimates=[]
            mxorder=[]
            print('start')
            for k in range(math.ceil(500/nNumDisjointrows)):
                for f in range(100):
                    #randomly generate # rows of indexes of length vnDisjointColumnIndex[z]
                    #from set of range(len(nDistjoinSet[z]))
                    vnIntSubSet=choices(range(len(nDistjoinSet[z])),weights=None,cum_weights=None,k=vnDisjointColumnIndex[z])
                    nIntSubSetSize=math.floor(vnDisjointColumnIndex[z]*self.npointspartition/self.ndatalength)-1
                    #set up tables of estimates of index (through c steps)
                    #initial estimate throug the matrix generated, first estimate will use initial
                    #estimate, this is to estimate index number 
                    mxInitialEstimate=self.estimate(nIntSubSetSize)
                    mxFirstEstimate=self.cstep(nDistjoinSet[z],math.floor(vnDisjointColumnIndex[z]*self.npointspartition/self.ndatalength),mxInitialEstimate)
                    mxSecondEstimate=self.cstep(nDistjoinSet[z],math.floor(vnDisjointColumnIndex[z]*self.npointspartition/self.ndatalength),[mxFirstEstimate[0],mxFirstEstimate[1]])
                    mxCandidateEstimates.append(mxSecondEstimate)
                    mxorder.append(np.linalg.det(mxSecondEstimate[1]))
             #sort and take top 10 estimates 
            mxCandidateEstimates=np.asarray(mxCandidateEstimates)
            mxorder=np.argsort(mxorder)
            tenordering=[]
            mxorder=mxorder.tolist()
            for i in range(10):
                tenordering.append(mxorder.index(i))
                candidate=mxCandidateEstimates[tenordering]
            mxEstimatesInsideSubsets.append(candidate)
        #flatten mxEstimatesInsideSubsets to  one dimensional
        mxEstimatesInsideSubsets = np.asarray(mxEstimatesInsideSubsets)
        mxEstimatesInsideSubsets=mxEstimatesInsideSubsets.flatten()
        #calculate c initial estimates for mean and cov using data generated
        #by using i index provided by mxEstimatesInsideSubsets
        #for second estimate, we use estimates from second 
        #initial estimate calculated by doing mean and cov calculations of 
        #each mxEstimatesInsideSubsets.
        CandidateEstimates=[]
        for i in mxEstimatesInsideSubsets:
            vnCurrentSubSet = []
            for j in i[2]:
                vnCurrentSubSet.append(self.data[j])
            mxInitialEstimate=[np.mean(vnCurrentSubSet,axis=0),np.cov(vnCurrentSubSet,rowvar=False)]
            mxFirstEstimate=self.cstep(self.data,self.npointspartition,mxInitialEstimate)
            mxSecondEstimate=self.cstep(self.data,self.npointspartition,mxFirstEstimate)
            CandidateEstimates.append(mxSecondEstimate)
        self.c_step_conv(CandidateEstimates, mxorder)

    def xMCD(self):
        #step 2: condiiton if p is equal to # of rows, return mean and cov of matrix
        if self.npointspartition == self.ndatalength:
            self.equalpartition()
            return
        sortedx = self.data[self.data[:,0].argsort()]
        #step 3: if it's one column data, return mean and variance after bootstrapping
        if self.numofcomponent == 1:
            self.onecomp(sortedx)
            return
        # step 4: if # of rows is less than 600,
        if self.ndatalength<=600:
            self.smalldata()
            return
        else:
            self.bigdata()
            return
        

#practice inputs to test accuracy 
mnPhosphorData = [(0.4, 53., 64.), (0.4, 23., 60.), (3.1, 19., 71.), (0.6, 34., 
   61.), (4.7, 24., 54.), (1.7, 65., 77.), (9.4, 44., 81.), (10.1, 
   31., 93.), (11.6, 29., 93.), (12.6, 58., 51.), (10.9, 37., 
   76.), (23.1, 46., 96.), (23.1, 50., 77.), (21.6, 44., 93.), (23.1, 
   56., 95.), (1.9, 36., 54.), (26.8, 58., 168.), (29.9, 51., 99.)]
a = fast_MCD(mnPhosphorData).xMCD()
# mnColemanData = ((3.83, 28.87, 7.2, 26.6, 6.19, 37.01), (2.89, 
#     20.1, -11.71, 24.4, 5.17, 26.51), (2.86, 69.05, 12.32, 25.7, 7.04,
#      36.51), (2.92, 65.4, 14.28, 25.7, 7.1, 40.7), (3.06, 29.59, 6.31,
#      25.4, 6.15, 37.1), (2.07, 44.82, 6.16, 21.6, 6.41, 33.9), (2.52, 
#     77.37, 12.7, 24.9, 6.86, 41.8), (2.45, 24.67, -0.17, 25.01, 5.78, 
#     33.4), (3.13, 65.01, 9.85, 26.6, 6.51, 41.01), (2.44, 9.99, -0.05,
#      28.01, 5.57, 37.2), (2.09, 12.2, -12.86, 23.51, 5.62, 
#     23.3), (2.52, 22.55, 0.92, 23.6, 5.34, 35.2), (2.22, 14.3, 4.77, 
#     24.51, 5.8, 34.9), (2.67, 31.79, -0.96, 25.8, 6.19, 33.1), (2.71, 
#     11.6, -16.04, 25.2, 5.62, 22.7), (3.14, 68.47, 10.62, 25.01, 6.94,
#      39.7), (3.54, 42.64, 2.66, 25.01, 6.33, 31.8), (2.52, 
#     16.7, -10.99, 24.8, 6.01, 31.7), (2.68, 86.27, 15.03, 25.51, 7.51,
#      43.1), (2.37, 76.73, 12.77, 24.51, 6.96, 41.01))
# mnColemanData=np.asarray(mnColemanData)
# b = fast_MCD(mnColemanData)
# b.xMCD()
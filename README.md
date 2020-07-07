# MCD-HMMUG

This Github Repo contains two project which were completed through master's educations. 
Both of them involve translating exisitng code in mathematica into python format. Please Note:
i do not claim to be expert in both of the subject matter and this is just for me to practice 
python.

## Minimum Covariance Determinant
  
Covariance is often used as an unbaised estimator of risk. However, convariance can still
suffer breakdowns as a result of few extreme data values. These values heavily influence 
the covariance and thus need to be eliminated at times. This project is a translation of 
the proposed method by Rousseeuw and Van Driessen (1999). For my code, the cases are 
separated in regards to data size.  
  
The three test cases are:
- Univariate
- Data size <= 600 
- Data Size > 600.  

Data should be cleaned before running the model.  The ouput of the data will be 
![MCDOUT](/pics/mcd.png).  
Please read Rousseeuw and Van Driessen (1999) if you wish to gain a better understanding of 
fast MCD algorithm 
## Univariate Gaussian Hidden Markov Model
  
The idea of Hidden markov model is that there are fininite number of discrete states   
we can be in. Similar to interconnected rooms. At any given time, theres a probability
we move into another room or stay in the same room. In a hidden markov model, the states(rooms)
are not known. The probability of a person ending up in another room is dependent on the room he 
was in at the previous discrete time.   
The basis of the Hidden markov model is a special version of EM algorithm(estimation, maximization) which involves 
forward recursion followed by backwards recursion， named Balm Welsh Algorithm.  
This algorithm is important to detech when market environments move from one state to another and can be used to 
take advantage of regime changes. The picture below shows a simplified version.  
![hmm](/pics/hmm.png)
  The function terminates if we either reached the maximum iteration(inputted variable) or the maximum likelihood function has 
converged. 
The termination function of the is  
The Output of the function is shown below.  
   
![huDOUT](/pics/hummg.png).  
The output might be confusing as i have intentional avoided going deep into the mathematics of the model and as a result, did not explain each of the
parameters that we see in the output.  

This is not a finished project, there is still more to be done including fixing >600 case for Fast MCD

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:43:54 2018

@author: Cole
"""

# Cole Grabowski
# This code corresponds to the completed solution for exercise 10 in biocomputing
# due on 11/14/18 on Statistical and Dynamic Modelling

#########################################
## Problem 1
# in lecture, we used maximum likelihood and a likelihood ratio test to complete
# a t-test. We can actually use a likelihood ratio test to compare two
# models as long as one model is a subset of the other model.
# For example, we can ask whether y is a hump-shaped vs linear function of 
# x by comparing a quadratic (a+bx+cx^2) vs linear (a+bx) model.
# Generate a script that evaluates which model is more appropriate
#  for the data in data.txt

# adapted from inclass_challenge_11_7.pyy
# general format for estimating statistical model parameters
# 1) load data
# 2) write a custom likelihood function
# 3) estimate parameter values by minimizing the negative log likelihood

## Import packages for part 1
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats
from plotnine import *

## Import packages for part 2
import scipy
import scipy.integrate as spint

## load data
data=pandas.read_csv('data.txt', sep=",", header=0)

# plot our observations
ggplot(data, aes(x='x',y='y'))+geom_point()+theme_classic()

## custom likelihood function for linear model
def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

## custom likelihood function for quadratic model
def nllike_quad(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    
    expected=B0+B1*obs.x+B2*(obs.x)*(obs.x)
    nll_quad=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll_quad

## estimate parameters by minimizing the negative log likelihood
initialGuess=numpy.array([1,1,1])
fit_linear=minimize(nllike,initialGuess,method='Nelder-Mead',options={'disp': True},args=data)

# fit is a variable that contains an OptimizeResult object
# attribute 'x' is a list of the most likely parameter values
print('Linear')
print(fit_linear.x)

# do the same for the quadratic model

## estimate parameters by minimizing the negative log likelihood
initialGuess_quad=numpy.array([1,1,1,1]) # rough guess doesnt matter so much
fit_quad=minimize(nllike_quad,initialGuess_quad,method='Nelder-Mead',options={'disp': True},args=data)

# fit is a variable that contains an OptimizeResult object
# attribute 'x' is a list of the most likely parameter values
print('Quadratic')
print(fit_quad.x)

######## Use Chi-squared model to give p-value (looking for p<0.05)
# coding a likelihood ratio test
# we have already found our two different likelihood function
# 
# now we need to calculate our test statustic, which is given by teststat below

# finally, we need to compare test statistic to a chi-squared distribution
# with degrees of freedom equal to the difference in numer of parameters between 
# our complex and simple models, and see wheether it is extreme relative to random chance
# looking for p value <0.05

teststat=2*(fit_linear.fun-fit_quad.fun) # calculate test statistic

df=len(fit_quad.x)-len(fit_linear.x)

p=1-stats.chi2.cdf(teststat,df) # calculate p value

print('')
print('P-value = ',p)

print('')


if (p<0.05):
    print('Quadratic Model is more appropriate than Linear Model')
else:
    print('Linear Model is more appropriate than Quadratic Model')

# since the p value is >>0.05, the Linear model is much better than the quadratic model
# as a sanity check, looking at the scatter plot of the data confirms visually that
# the model is much more suited to being described by a linear model
    

# fits are determined  by minimizing the negative log likelihood
# For the linear fit, python was able to converge to a solution after 127 iterations
# of the following form:
#  Linear: 16.969+4.4825*x, with an error of 18.0038
#
# For the quadratic fit, the results showed that a quadratic model for this data
# set is not as appropriate as the linear model.
# The quadratic model solving exceeded the maximum number of function evaluations,
# meaning that no solutuion converged
#
# Quadratic: 1.581E1+4.627*x-2.8447E-3, with an error of 18.0





#############################################
## Problem 2
# A classic model of competition between two species was developed by Lotka
# and Volterra. This model has two state variables described by 2 diff eqs
#
# dN_1/dt=R_1(1-N_1*alpha_11-N_2*alpha_12)*N_1
# dN_2/dt=R_2(1-N_2*alpha_22-N_1*alpha_21)*N_2
#
# The criteria for coexistence of two species in the Lotka Volterra 
# competition model is ...
#
# alpha_12 < alpha_11 and alpha_21 < alpha_22
#
# Generate a script that uses three or more model simulations to demonstrate
# the validity of these criteria for coexistence

# code very closely resembles that used during SOLUTION_inclass_challenge_11_12.py
# REMEMBER alpha=1/k


def LotVolSim(y,t0,R_1,R_2,alpha_11,alpha_12,alpha_21,alpha_22):
    N_1=y[0]
    N_2=y[1]
    
    dNdt_1=R_1*(1-N_1*alpha_11-N_2*alpha_12)*N_1
    dNdt_2=R_2*(1-N_2*alpha_22-N_1*alpha_21)*N_2
    
    return [dNdt_1,dNdt_2]

#######################
# case 1
times=range(0,100) # passed to LotVolSim as t0
y0=[0.1,0.1] # passed to LotVolSim as y in this case, you can call it y0=y0 since y is just understood as the first argument given and called


# R represents rate of growth, has to be between 0 and 1
R_1= 0.4
R_2=0.3

# alpha represents competition between populations 1 and 2
## note: Lotka Volterra Model requires alphas chosen such that...
# alpha_12 < alpha_11 and alpha_21 < alpha_22
alpha_11=0.2
alpha_12=0.1
alpha_21=0.1
alpha_22=0.2

params=(R_1,R_2,alpha_11,alpha_12,alpha_21,alpha_22) # params given in the order of LotVolSim function call above
sim=spint.odeint(func=LotVolSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
a=ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()
print(a)

# analysis of plot:
# since R_1 is slightly greateer than R_2, it makes sense that population 1 
# represented by the black curve on the first plot for part 2 has
# a slightly higher growth rate than the red line for population 2.
# Since the alpha_12 and alpha_21 values are the same in this case, it also 
# makes sense that the final values should converge as t->100

# In this model, since  alpha_12 < alpha_11 and alpha_21 < alpha_22, the populations
# can coexist, shown by the convergence of the red and black lines over time


############################
# case 2
times=range(0,500) 
y0=[0.05,0.25]

# R represents rate of growth, has to be between 0 and 1
R_1= 0.9
R_2=0.2

# alpha represents competition between populations 1 and 2
## note: Lotka Volterra Model requires alphas chosen such that...
# alpha_12 < alpha_11 and alpha_21 < alpha_22
alpha_11=0.1
alpha_12=0.5
alpha_21=0.6
alpha_22=0.1

params=(R_1,R_2,alpha_11,alpha_12,alpha_21,alpha_22) # params given in the order of LotVolSim function call above
sim=spint.odeint(func=LotVolSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
b=ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()
print(b)

# analysis of plot
# Since the starting population for N2 is higher than N1, this makes sense based on the 
# red line starting higher than the black line at t0
# Since the growth rate for population 1 is much higher than for population 2,
# it makes sense that the slope of the black line at the beginning is much steeper than that
# of the red line. 

# In this case, the alpha values were chosen to ignore the criteria that
# # alpha_12 < alpha_11 and alpha_21 < alpha_22.
# Since these criteria were ignored, the species cannot coexist, shown by the red line
# of population dropping off to zero

#############################
# case 3
times=range(0,100)
y0=[0.6,0.1]

# R represents rate of growth, has to be between 0 and 1
R_1= 0.1
R_2=0.85

# alpha represents competition between populations 1 and 2
## note: Lotka Volterra Model requires alphas chosen such that...
# alpha_12 < alpha_11 and alpha_21 < alpha_22
alpha_11=0.1
alpha_12=0.5
alpha_21=0.6
alpha_22=0.2

params=(R_1,R_2,alpha_11,alpha_12,alpha_21,alpha_22) # params given in the order of LotVolSim function call above
sim=spint.odeint(func=LotVolSim,y0=y0,t=times,args=params)
simDF=pandas.DataFrame({"t":times,"Population 1":sim[:,0],"Population 2":sim[:,1]})
c=ggplot(simDF,aes(x="t",y="Population 1"))+geom_line()+geom_line(simDF,aes(x="t",y="Population 2"),color='red')+theme_classic()
print(c)

# in this case, the criteria that
# alpha_12 < alpha_11 and alpha_21 < alpha_22
# was also ignored, but attempts were made by adjusting different parameters to
# allow the species to coexist.

# For example, Population 1 started with a high population but a low growth rate,
# whereas population 2 started with a low population and a high growth rate. The 
# alpha values between the populations were similar but still different enough to 
# attempt to increase the longevity of coexistence.

# With all of these efforts, Population 1 and 2 did coexist for longer periods of time
# than in Case 2 for example, but by not meeting the criteria that
# alpha_12 < alpha_11 and alpha_21 < alpha_22
# this coexistence was shortlived and population 2 ended up winning out.









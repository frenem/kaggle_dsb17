from math import exp
from math import floor
from random import randint

class Unit:
  
  def __init__(self,value=0,grad=0):
    self.value = value
    self.grad  = grad
    
class MultiplyGate:
  
  def forward(self,u0,u1):
    self.u0 =u0 
    self.u1 =u1
    self.utop = Unit(u0.value * u1.value,0.0)
    return self.utop
  
  def backward(self):
    self.u0.grad += self.u1.value * self.utop.grad  
    self.u1.grad += self.u0.value * self.utop.grad
    
class addGate:
  
  def forward(self,u0,u1):
    self.u0 =u0 
    self.u1 =u1
    self.utop = Unit(u0.value + u1.value,0.0)
    return self.utop

  def backward(self):
    self.u0.grad += 1 * self.utop.grad  
    self.u1.grad += 1 * self.utop.grad

    
# Definition of Circuit class 
# Implementation of the Neuron 
# Computes ax +by +c and the associated gradient

class Circuit:
  def __init__(self):
    self.mulg0 = MultiplyGate()
    self.mulg1 = MultiplyGate()
    self.addg0 = addGate()
    self.addg1 = addGate()

  def forward(self,a,b,c,x,y):
    self.ax= self.mulg0.forward(a,x)
    self.by= self.mulg1.forward(b,y)
    self.axpby = self.addg0.forward(self.ax,self.by)
    self.axpbypc = self.addg1.forward(self.axpby,c)
    return self.axpbypc
  
  def backward(self,gradient_top):
    self.axpbypc.grad = gradient_top
    self.addg1.backward()
    self.addg0.backward()
    self.mulg1.backward()
    self.mulg0.backward()


#Definition of the SVM class
class SVM:
    def __init__(self):
        # Initilize the 
        self.a = Unit(1.0,0.0)
        self.b = Unit(-2.0,0.0)
        self.c = Unit(-1.0,0.0)
  
        self.circuit = Circuit()
    
    def forward(self,x,y):
        self.unit_out = self.circuit.forward(self.a,self.b,self.c,x,y)
        return self.unit_out
    
    def backward(self,label):
        #Reset_pull
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0
        self.pull   = 0.0
    
        if label == 1 and self.unit_out.value < 1:
            self.pull = 1
            
        if label == -1 and self.unit_out.value > -1:
            self.pull = -1
        
        self.circuit.backward(self.pull)
    
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value
        
    def learnFrom(self,x,y,l):
        self.forward(x,y)
        self.backward(l)
        self.parameterUpdate()
    
    def parameterUpdate(self):
        self.step_size = 0.1
        self.a.value += self.step_size * self.a.grad
        self.b.value += self.step_size * self.b.grad
        self.c.value += self.step_size * self.c.grad
        
        print ' ******** ********** **********'  
        print 'New Value of a : {} '.format(self.a.value) 
        print 'New Value of b : {} '.format(self.b.value) 
        print 'New Value of c : {} '.format(self.c.value) 
        print ' ******** ********** **********'  

data = [[1.2,0.7],[-0.3,0.5],[-3,-1],[0.1,1.0],[3.0,1.1],[2.1,-3]]
labels  = [1,-1,1,-1,-1,1]
  
svm = SVM()

def evalTrainingAccuracy():
    num_correct=0
    for i in range(0,len(data)):
        x = Unit(data[i][0],0.0)
        y = Unit(data[i][1],0.0)
        trueLabel = labels[i]

        predicted_label = 1 if svm.forward(x,y).value > 0 else -1
        
        if predicted_label == trueLabel:
            num_correct += 1
           
    #return 'Number of correct guesses ' + str(num_correct)
    print '***** Training Result *****' 
    print '* Number of correct guesses : {}'.format(num_correct) 
    print '* Percentage of correct guesses : {}%'.format(floor(num_correct*100.00/len(data)),3)
    print '***** Training Result (END) *****' 
    print ' ' 

for iter in range(0,1000):

    # Select a random number 
    i = int(floor(randint(0, len(data)-1)))

   # print 'Itearation Number : {}'.format(iter)
    
    x = Unit(data[i][0],0.0)
    y = Unit(data[i][1],0.0)
    label = labels[i]
    
    if iter == 999 or iter == 0: 
    
        print ' '
        print ' Iteration Number : {}'.format(iter)  
        print '***** Initial Values *****'
        print 'a : {}'.format(svm.a.value)
        print 'b : {}'.format(svm.b.value)
        print 'c : {}'.format(svm.c.value)
        print 'X : {}'.format(x.value)
        print 'Y : {}'.format(y.value)
        print 'label : {}'.format(label)
        print '**************************'
        print ' ' 
        
    svm.learnFrom(x,y,label)
    
    evalTrainingAccuracy()

    

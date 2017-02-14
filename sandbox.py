from math import exp

class Unit:
  
  def __init__(self,value=0,grad=0):
    self.value = value
    self.grad = grad
    
    
class MultiplyGate:
  
  def forward(self,u0,u1):
    self.u0 =u0 
    self.u1 =u1
    self.utop = Unit(u0.value * u1.value,0.0)
  
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
    
    
class sigmoidGate:
  
  def sig(a):
    return 1/(1+exp(-x))
    
  def forward(self,u0,u1):
    self.u0 =u0 
    self.utop = Unit(sig(self.u0.value),0.0)
    return self.utop
    
  def backward(self):
    s = sig(self.u0.value)   
    self.u0.grad +=  (s * (1-s)) * self.utop.grad
    

def forwardMultiplyGate(x,y):
  return x*y 
  
x= -2 
y= 3
  
out = forwardMultiplyGate(x,y);

x_gradient = y
y_gradient = x

step_size = 0.01

x+= step_size * x_gradient
y+= step_size * y_gradient

out_new = forwardMultiplyGate(x,y);

print out,out_new

import torch
from SphericalCoordinates import *

from RBF import *
N,D_in,D_out = 10,3,2
torch.random.manual_seed(50)

L = RBFLay(D_in,D_out).to('cuda')



from torch.autograd import gradcheck
linear = L.apply
x = torch.randn(N,D_in,dtype=torch.double,requires_grad=True).to('cuda:0')
#w = 10*torch.randn(D_out,D_in,dtype=torch.double,requires_grad=True).to('cuda:0')
#b = 10*torch.randn(D_out,1,dtype=torch.double,requires_grad=True).to('cuda:0')
#rho = 0.01*torch.abs(torch.randn(D_out,1,dtype=torch.double,requires_grad=True).to('cuda:0'))


test = gradcheck(L, x, eps=1e-5, atol=1e-6)
print(test)

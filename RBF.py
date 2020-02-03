
import torch
import pydevd




class RBFFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,w,b,rho):
        inter = (torch.mm(w,input)+b.t())
        y = torch.exp(-rho.t()*inter**2)
        ctx.save_for_backward(y,w,b,rho,inter,input)
        return y


    @staticmethod
    def backward(ctx, grad_output):
        #pydevd.settrace(suspend=False, trace_only_current_thread=True)
        y, w, b, rho, inter,input = ctx.saved_tensors
        grad_input = -torch.mm(w.t(),2 * rho.t() * y * inter*grad_output)
        grad_w = -torch.mm((2 * rho.t() * y * inter*grad_output),input.t())
        grad_b = -torch.sum((2 * rho.t() * y * inter*grad_output),dim=1).reshape((1,-1))
        grad_rho = -torch.sum((y * inter**2*grad_output),dim=1).reshape((1,-1))


        return grad_input,grad_w,grad_b,grad_rho# grad_w,grad_b,grad_rho


class RBFLay(torch.nn.Module):
    def __init__(self,N_input,N_output):
        super(RBFLay, self).__init__()
        self.N_input = N_input
        self.N_output = N_output

        self.w = torch.nn.Parameter(torch.Tensor(N_output, N_input).float())
        self.b = torch.nn.Parameter(torch.Tensor(1,N_output).float())
        self.rho = torch.nn.Parameter(torch.Tensor(1,N_output).float())
        self.w.data.uniform_(-0.1, 0.1)
        self.b.data.uniform_(-0.1, 0.1)
        self.rho.data.uniform_(0.0, 0.5)

    def forward(self, input):
        return RBFFunc.apply(input.t(),self.w,self.b,self.rho).t()


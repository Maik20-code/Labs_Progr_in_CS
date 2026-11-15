import numpy as np

class Module(object):
    def __init__(self):
        self.output = none
        self.gradinput = none
        self.training = true

    def forward(self, input):
        """
        takes an input object, and computes the corresponding output of the module.
        """
        return self.updateoutput(input)

    def backward(self, input, gradoutput):
        """
        performs a backpropagation step through the module, with respect to the given input.
        """
        self.updategradinput(input, gradoutput)
        self.accgradparameters(input, gradoutput)
        return self.gradinput

    def updateoutput(self, input):
        """
        computes the output using the current parameter set of the class and input.
        """
        pass

    def updategradinput(self, input, gradoutput):
        """
        computing the gradient of the module with respect to its own input.
        """
        pass

    def accgradparameters(self, input, gradoutput):
        """
        computing the gradient of the module with respect to its own parameters.
        """
        pass

    def zerogradparameters(self):
        """
        zeroes gradparams variable if the module has params.
        """
        pass

    def getparameters(self):
        """
        returns a list with its parameters.
        """
        return []

    def getgradparameters(self):
        """
        returns a list with gradients with respect to its parameters.
        """
        return []

    def train(self):
        """
        sets training mode for the module.
        """
        self.training = true

    def evaluate(self):
        """
        sets evaluation mode for the module.
        """
        self.training = false

    def __repr__(self):
        """
        pretty printing.
        """
        return "module"


class Sequential(Module):
    """
    This class implements a container, which processes input data sequentially.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)

    def updateOutput(self, input):
        y = input
        for m in self.modules:
            y = m.forward(y)
        self.output = y
        return self.output

    def backward(self, input, gradOutput):
        t = []  # len(t) eq len(self.modules)+1
        y = input
        t.append(y)
        for m in self.modules:
            y = m.forward(y)
            t.append(y)

        y = gradOutput
        for i in range(len(self.modules)):
            k = len(self.modules) - i - 1
            y = self.modules[k].backward(t[k], y)
        self.gradInput = y
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules[x]

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()


class Linear(Module):
    """
    A module which applies a linear transformation.
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        stdv = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, input):
        # y = X * W^T + b
        self.output = np.dot(input, self.W.T) + self.b
        return self.output

    def updateGradInput(self, input, gradOutput):
        # dL/dX = dL/dY * W
        self.gradInput = np.dot(gradOutput, self.W)
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        # dL/dW = (dL/dY)^T * X
        self.gradW += np.dot(gradOutput.T, input)
        # dL/db = sum(dL/dY)
        self.gradb += np.sum(gradOutput, axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' % (s[1], s[0])
        return q


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        shifted_input = np.subtract(input, input.max(axis=1, keepdims=True))
        exp_input = np.exp(shifted_input)
        self.output = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.output

    def updateGradInput(self, input, gradOutput):
        # dL/dx = p * (dL/dy - sum(dL/dy * p))
        sum_grad_out_p = np.sum(gradOutput * self.output, axis=1, keepdims=True)
        self.gradInput = self.output * (gradOutput - sum_grad_out_p)
        return self.gradInput

    def __repr__(self):
        return "SoftMax"


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, input):
        # start with normalization for numerical stability
        shifted_input = np.subtract(input, input.max(axis=1, keepdims=True))
        log_sum_exp = np.log(np.sum(np.exp(shifted_input), axis=1, keepdims=True))
        self.output = shifted_input - log_sum_exp
        return self.output

    def updateGradInput(self, input, gradOutput):
        softmax = np.exp(self.output)
        self.gradInput = gradOutput - np.sum(gradOutput, axis=1, keepdims=True) * softmax
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        if self.training:
            batch_mean = np.mean(input, axis=0)
            batch_variance = np.var(input, axis=0)

            if self.moving_mean is None:
                self.moving_mean = batch_mean.copy()
                self.moving_variance = batch_variance.copy()
            else:
                self.moving_mean = (self.moving_mean * self.alpha + batch_mean * (1. - self.alpha))
                self.moving_variance = (self.moving_variance * self.alpha + batch_variance * (1. - self.alpha))

            self.output = (input - batch_mean) / np.sqrt(batch_variance + self.EPS)
        else:
            self.output = (input - self.moving_mean) / np.sqrt(self.moving_variance + self.EPS)
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            batch_size = input.shape[0]
            batch_mean = np.mean(input, axis=0)
            batch_variance = np.var(input, axis=0)

            x_centered = input - batch_mean
            inv_std = 1. / np.sqrt(batch_variance + self.EPS)

            self.gradInput = (inv_std / batch_size) * (
                batch_size * gradOutput - np.sum(gradOutput, axis=0)
                - x_centered * (inv_std ** 2) * np.sum(gradOutput * x_centered, axis=0)
            )
        else:
            self.gradInput = gradOutput / np.sqrt(self.moving_variance + self.EPS)

        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        self.mask = None

    def updateOutput(self, input):
        if self.training:
            self.mask = (
                np.random.binomial(1, 1. - self.p, size=input.shape) / (1. - self.p)
            )
            self.output = input * self.mask
        else:
            self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask
        else:
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput, input > 0)
        return self.gradInput

    def __repr__(self):
        return "ReLU"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()
        self.slope = slope

    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, gradOutput, gradOutput * self.slope)
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


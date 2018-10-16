# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:16:32 2018

@author: admin
"""
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch as t
import PIL
import numpy as np
from torch.autograd import Variable
style_layers=[1,6,11,20,25]
content_layers=[21]
loss_layers=[1,6,11,20,25,21]
'''加载预训练vgg模型'''
vgg=models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad=False
vgg.cuda()
'''加载图片,图片转发为tensor，并且标准化'''
prep=transforms.Compose([transforms.Resize(512),transforms.ToTensor(),transforms.Lambda(lambda x: x[t.LongTensor([2,1,0])]),transforms.Normalize([0.407,0.457,0.485],[1,1,1]),transforms.Lambda(lambda x:x.mul_(255)),])
postpa=transforms.Compose([transforms.Lambda (lambda x :x.mul_(1./255)),transforms.Normalize([-0.407,-0.457,-0.483],[1,1,1]),transforms.Lambda(lambda x:x[t.LongTensor([2,1,0])],)])
postpb=transforms.Compose([transforms.ToPILImage()])
'''限制图像范围为（0，1）再通过postpb函数把tensor变为pilimg格式'''
def postp(tensor):
    t=postpa(tensor)
    t[t>1]=1
    t[t<0]=0
    img=postpb(t)
    return img
'''导入图片函数'''
def image_loader (image_name):
    image=PIL.Image.open(image_name)
    image=Variable(prep(image))
    image=image.unsqueeze(0)
    return image
'''加入你需要生成图像的路径替代下面的c:\\与c:\\,style_img是风格图画（例如梵高的作品），content_img是的需要加入画风的图画地址'''
style_img=image_loader('./stye.jpg').cuda()
content_img=image_loader('./content.jpg').cuda()
#print(content_img.size())
opt_img=Variable(content_img.data.clone(),requires_grad=True).cuda()
'''计算gram矩阵'''
class GramMatrix(nn.Module):
    def forward (self,input):
        b,c,h,w=input.size()
        features=input.view(b,c,h*w)
        gram_matrix=t.bmm(features,features.transpose(1,2))
        gram_matrix.div_(h*w)
        return gram_matrix
'''创建 hook 提取vgg中特定卷积层中的fetures'''
class layerActivations():
    features=[]
    def __init__(self,model,layer_nums):
        self.hooks=[]
        for layer_num in layer_nums:
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))
    def hook_fn(self,module,input,output):
        self.features.append(output)
    def remove(self):
        for hook in self.hooks:
            hook.remove()        
def extract_layers(layers,img,model=None):
    la =layerActivations(model,layers)
    la.features=[]
    _=model(img)
    la.remove()
    return la.features
'''创建风格损失函数'''
class Styloss(nn.Module):
    def forward(self,inputs,targets):
        out=nn.MSELoss()(GramMatrix()(inputs),targets)
        return (out)
'''提取目标特征，初始化生成图像特征（默认等于目标内容图），初始化损失权值'''
content_targets=extract_layers(content_layers,content_img,model=vgg)
style_targets=extract_layers(style_layers,style_img,model=vgg)
content_targets=[t.detach()for t in content_targets]
style_targets=[GramMatrix()(t).detach()for t in style_targets]
style_weights=[1e3/n**2 for n in [64,128,256,512,512]]
content_weights=[1e0]
loss=nn.MSELoss()
'''加入优化函数lbfgs，初始最大迭代次数与计算总损失（mse的weight sum）'''
optimizer=t.optim.LBFGS([opt_img])
weights=style_weights+content_weights
max_iter=700
show_iter=50
n_iter=[0]
loss_fns=[Styloss()]*len(style_layers)+[nn.MSELoss()]*len(content_layers)
targets=style_targets+content_targets
layer_loss=[]
while n_iter[0]<= max_iter:
    def closure():
        optimizer.zero_grad()
        out =extract_layers(loss_layers,opt_img,model=vgg)
        layer_loss=[weights[a]*loss_fns[a](A,targets[a])for a,A in enumerate(out)]
        loss=sum(layer_loss)
        loss.backward()
        n_iter[0]+=1
        if n_iter[0]%show_iter==(show_iter-1):
            print('iteration:%d,loss:%f'%(n_iter[0]+1,loss.data[0]))
        return loss
    optimizer.step(closure)
print(type(opt_img.data),opt_img.data.size())    
img=postp(opt_img.data.squeeze(0).cpu())
img.save('./test.jpg')
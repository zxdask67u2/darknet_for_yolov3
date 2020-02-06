import torch
import torch.nn as nn
from torch.nn import functional
class con_bn_leaky(nn.Module):
    def __init__(self,in_ch,out_ch,k,stride,padd):
        super(con_bn_leaky, self).__init__()
        self.cbl=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,k,stride,padd),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        ).cuda()
    def forward(self, x):
        x=self.cbl(x)
        print(x.size())
        return x

class con_bn_leak_5set(nn.Module): #
    def __init__(self,in_ch,out_ch):
        super(con_bn_leak_5set, self).__init__()
        self.convset = nn.Sequential(
            con_bn_leaky(in_ch, out_ch, 1, 1, 0),
            con_bn_leaky(out_ch, out_ch, 3, 1, 1),
            con_bn_leaky(out_ch, out_ch * 2, 1, 1, 0),
            con_bn_leaky(out_ch * 2, out_ch * 2, 3, 1, 1),
            con_bn_leaky(out_ch * 2, out_ch, 1, 1, 0)
        ).cuda()
    def forward(self, x):
        return self.convset(x)

class upsampling(nn.Module):
    def __init__(self):
        super(upsampling, self).__init__()
    def forward(self, x):
        return nn.functional.interpolate(x,scale_factor=2).cuda()#放大两倍

class down_sample(nn.Module):
    def __init__(self,in_c,out_c):
        super(down_sample, self).__init__()
        self.inc=in_c
        self.outc=out_c
    def forward(self, x):#变为1/2
        return con_bn_leaky(self.inc,self.outc,3,2,1)(x)

class Res_block(nn.Module):
    def __init__(self,in_ch):
        super(Res_block, self).__init__()
        self.res=nn.Sequential(
            con_bn_leaky(in_ch,in_ch//2,1,1,0),
            con_bn_leaky(in_ch//2,in_ch,3,1,1)
        ).cuda()
    def forward(self, x):
        return x+self.res(x)

class Main_net(nn.Module):
    def __init__(self):
        super(Main_net, self).__init__()

        self.d_52=nn.Sequential(
            con_bn_leaky(3,32,3,1,1),#416*416
            con_bn_leaky(32,64,3,2,1),#208*208

            #res_unit1
            con_bn_leaky(64, 32, 1, 1, 0),#1*1卷积尺度不变 208
            con_bn_leaky(32,64,3,1,1),#尺度不变 208
            Res_block(64),

            down_sample(64,128),  #缩小2分之1 -- 104

            #res_unit2
            con_bn_leaky(128, 64, 1, 1, 0),
            con_bn_leaky(64, 128, 3, 1, 1),
            Res_block(128),

            con_bn_leaky(128, 64, 1, 1, 0),
            con_bn_leaky(64, 128, 3, 1, 1),
            Res_block(128),

            down_sample(128, 256),  # 52

            #res_unit8    大小一直不变  52到底
            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),

            con_bn_leaky(256, 128, 1, 1, 0),
            con_bn_leaky(128, 256, 3, 1, 1),
            Res_block(256),#   52
        )
        self.d_26=nn.Sequential(
            down_sample(256,512), #26*26

            #res_unit8
            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

            con_bn_leaky(512, 256, 1, 1, 0),
            con_bn_leaky(256, 512, 3, 1, 1),
            Res_block(512),

        )

        self.d_13=nn.Sequential(
            down_sample(512,1024),

            #res_unit4
            con_bn_leaky(1024,512,1,1,0),
            con_bn_leaky(512, 1024, 3, 1, 1),
            Res_block(1024),

            con_bn_leaky(1024, 512, 1, 1, 0),
            con_bn_leaky(512, 1024, 3, 1, 1),
            Res_block(1024),

            con_bn_leaky(1024, 512, 1, 1, 0),
            con_bn_leaky(512, 1024, 3, 1, 1),
            Res_block(1024),

            con_bn_leaky(1024, 512, 1, 1, 0),
            con_bn_leaky(512, 1024, 3, 1, 1),
            Res_block(1024),
        )

        self.conv_set_13=con_bn_leak_5set(1024,512)









    def forward(self, x):
        out=self.d_52(x)
        out=self.d_26(out)
        out=self.d_13(out)
        print('输出',out.size())
        return out



if __name__ == '__main__':
    device = torch.device("cuda:0")
    x=torch.rand([2,3,416,416]).to(device)
    m=Main_net().to(device)
    out=m(x)




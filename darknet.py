import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
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

class draknet53(nn.Module):
    def __init__(self,class_num):
        super(draknet53, self).__init__()
        self.out_filter_ch=3*(class_num+5)

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
        #13*13输出部分------------------------------------
        self.conv_set_13=con_bn_leak_5set(1024,512)
        self.out1_conv=nn.Sequential(
            con_bn_leaky(512,512,3,1,1),
            nn.Conv2d(512,self.out_filter_ch,1,1,0)
        )

        #13向26con_set传播部分
        self.con13_26=nn.Sequential(
            con_bn_leaky(512,256,1,1,0),
            upsampling()
        )    #256*16*16

        #26*26输出部分，------------------------------
        # 输入：经过concat后（256+512）=768
        self.conv_set_26=nn.Sequential(
            con_bn_leak_5set(768,512)
        )
        self.out2_conv=nn.Sequential(
            con_bn_leaky(512,512,3,1,1),
            nn.Conv2d(512,self.out_filter_ch,1,1,0)
        )

        #26向52转换部分
        self.con26_52=nn.Sequential(
            con_bn_leaky(512,256,1,1,0),
            upsampling()
        )

        # 52*52输出部分，------------------------------
        # 输入：经过concat后（256+256）=512  正合适
        self.conv_set_52=nn.Sequential(
            con_bn_leak_5set(512,512)
        )
        self.out3_conv=nn.Sequential(
            con_bn_leaky(512,512,3,1,1),
            nn.Conv2d(512,self.out_filter_ch,1,1,0)
        )











    def forward(self, x):
        x52=self.d_52(x)
        x26=self.d_26(x52)
        x13=self.d_13(x26)

        x13_=self.conv_set_13(x13)
        out1=self.out1_conv(x13_)#13*13输出----------------

        x_13_26=self.con13_26(x13_)
        x_26_in=torch.cat((x_13_26,x26),dim=1)
        x_26_=self.conv_set_26(x_26_in)
        out2=self.out2_conv(x_26_)

        x_52_=self.con26_52(x_26_)
        x_52_=torch.cat((x_52_,x52),dim=1)
        out3=self.conv_set_52(x_52_)
        out3=self.out3_conv(out3)



        return out1,out2,out3



if __name__ == '__main__':
    import time
    device = torch.device("cuda:0")

    m=draknet53(1).to(device)
    m.eval()
    with torch.no_grad():    #防止累加显存溢出
        x=torch.rand([2,3,416,416]).to(device)
        for i in range(3):
            t1=time.time()
            out1,out2,out3=m(x)
            print(time.time()-t1)
            print(out1.size(),out2.size(),out3.size())




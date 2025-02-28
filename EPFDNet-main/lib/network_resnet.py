
from lib.module import*

class Network(nn.Module):
    def __init__(self, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # if self.training:
        # self.initialize_weights()

        self.rfb1_1 = RFB_modified(256, 64)
        self.rfb2_1 = RFB_modified(512, 64)
        self.rfb3_1 = RFB_modified(1024, 64)
        self.rfb4_1 = RFB_modified(2048, 64)
        self.conv = Conv1x1(2048, 64)
        self.block = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1))

        self.upsample = cus_sample
        self.edge = ESM()
        self.sem1 = SEM(64, 64)
        self.sem2 = SEM(64, 64)
        self.sem3 = SEM(64, 64)
        self.sem4 = SEM(64, 64)

        self.aspp = ASPP(64)

        self.fam1 = FAM(64)
        self.fam2 = FAM(64)
        self.fam3 = FAM(64)

        self.reduce2 = Conv1x1(64, 128)
        self.reduce3 = Conv1x1(64, 256)

        self.predictor1 = nn.Conv2d(64, 1, 3, padding=1)
        self.predictor2 = nn.Conv2d(128, 1, 3, padding=1)
        self.predictor3 = nn.Conv2d(256, 1, 3, padding=1)
        self.predictor4 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        #x1, x2, x3, x4 = self.resnet(x)
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        x0 = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x0)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        x1_rfb = self.rfb1_1(x1)  # channel -> 64
        x2_rfb = self.rfb2_1(x2)  # channel -> 64
        x3_rfb = self.rfb3_1(x3)  # channel -> 64
        x4_rfb = self.rfb4_1(x4)  # channel -> 64

        x2_r = -1 * (torch.sigmoid(x2_rfb)) + 1
        x2_r = self.block(x2_r)

        x3_r = -1 * (torch.sigmoid(x3_rfb)) + 1
        x3_r = self.block(x3_r)

        x4_r = -1 * (torch.sigmoid(x4_rfb)) + 1
        x4_r = self.block(x4_r)

        x2a = self.sem2(x2_rfb, x2_r)
        x2a = self.upsample(x2a, scale_factor=2)
        x3a = self.sem3(x3_rfb, x3_r)
        x3a = self.upsample(x3a, scale_factor=4)
        x4a = self.sem4(x4_rfb, x4_r)
        x4a = self.upsample(x4a, scale_factor=8)
        x41 = self.conv(x4)
        x41 = self.upsample(x41, scale_factor=8)
        x4 = self.aspp(x41)

        edge = self.edge(x1_rfb)
        edge_att = torch.sigmoid(edge)

        x34 = self.fam1(x4a, edge_att, x4)
        x234 = self.fam2(x3a, edge_att, x34)
        x1234 = self.fam3(x2a, edge_att, x234)

        x34 = self.reduce3(x34)
        x234 = self.reduce2(x234)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=4, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=4, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        #oe = self.predictor4(edge)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe


if __name__ == '__main__':
    import numpy as np
    from time import time
    net = Network(imagenet_pretrained=False)
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352)
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        start = time()
        y = net(dump_x)
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
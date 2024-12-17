import torch
import torch.nn as nn
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import torch.nn.functional as F
import cv2
import torch.nn.utils as nn_utils
import numpy as np
from networks.xception import Xception
from models import clip
import resnet18_gram as resnet
import matplotlib.pyplot as plt


# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


def get_xcep_state_dict(pretrained_path='pretrained/xception-b5690688.pth'):
    # load Xception
    state_dict = torch.load(pretrained_path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
    return state_dict


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


# FAD Module
class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size * 2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 299, 299]
        return out


# LFS Module
class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1),
                                         requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList(
            [Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i + 1), norm=True) for i in
             range(M)])

    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8) / 2) + 1
        assert size_after == 112

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)  # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2, 3, 4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)  # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out


class SA_layer(nn.Module):
    def __init__(self, dim=256, head_size=8):
        super(SA_layer, self).__init__()
        self.mha=nn.MultiheadAttention(dim, head_size)
        self.ln1=nn.LayerNorm(dim)
        self.fc1=nn.Linear(dim, dim)
        self.ac=nn.ReLU()
        self.fc2=nn.Linear(dim, dim)
        self.ln2=nn.LayerNorm(dim)

    def forward(self, x):
        batch_size, len_size, fea_dim=x.shape
        x=torch.transpose(x,1,0)
        y,_=self.mha(x,x,x)
        x=self.ln1(x+y)
        x=torch.transpose(x,1,0)
        x=x.reshape(batch_size*len_size, fea_dim)
        x=x+self.fc2(self.ac(self.fc1(x)))
        x=x.reshape(batch_size,len_size, fea_dim)
        x=self.ln2(x)
        return x


class COOI(): # Coordinates On Original Image
    def __init__(self):
        self.stride=32
        self.cropped_size=224
        self.score_filter_size_list=[[3,3],[2,2]]
        self.score_filter_num_list=[3,3]
        self.score_nms_size_list=[[3,3],[3,3]]
        self.score_nms_padding_list=[[1,1],[1,1]]
        self.score_corresponding_patch_size_list=[[224, 224], [112, 112]]
        self.score_filter_type_size=len(self.score_filter_size_list)

    def get_coordinates(self, fm, scale):
        with torch.no_grad():
            batch_size, _, fm_height, fm_width=fm.size()
            scale_min=torch.min(scale, axis=1, keepdim=True)[0].long()
            scale_base=(scale-scale_min).long()//2 #torch.div(scale-scale_min,2,rounding_mode='floor')
            input_loc_list=[]
            for type_no in range(self.score_filter_type_size):
                score_avg=nn.functional.avg_pool2d(fm, self.score_filter_size_list[type_no], stride=1) #(7,2048,5,5), (7,2048,6,6)
                score_sum=torch.sum(score_avg, dim=1, keepdim=True) #(7,1,5,5), (7,1,6,6) #since the last operation in layer 4 of the resnet50 is relu, thus the score_sum are greater than zero
                _,_,score_height,score_width=score_sum.size()
                patch_height, patch_width=self.score_corresponding_patch_size_list[type_no]

                for filter_no in range(self.score_filter_num_list[type_no]):
                    score_sum_flat=score_sum.view(batch_size, -1)
                    value_max,loc_max_flat=torch.max(score_sum_flat, dim=1)
                    #loc_max=torch.stack((torch.div(loc_max_flat,score_width,rounding_mode='floor'), loc_max_flat%score_width), dim=1)
                    loc_max=torch.stack((loc_max_flat//score_width, loc_max_flat%score_width), dim=1)
                    top_patch=nn.functional.max_pool2d(score_sum, self.score_nms_size_list[type_no], stride=1, padding=self.score_nms_padding_list[type_no])
                    value_max=value_max.view(-1,1,1,1)
                    erase=(top_patch!=value_max).float() # due to relu operation, the value are greater than 0, thus can be erase by multiply by 1.0/0.0
                    score_sum=score_sum*erase

                    # location in the original images
                    loc_rate_h=(2*loc_max[:,0]+fm_height-score_height+1)/(2*fm_height)
                    loc_rate_w=(2*loc_max[:,1]+fm_width-score_width+1)/(2*fm_width)
                    loc_rate=torch.stack((loc_rate_h, loc_rate_w), dim=1)
                    loc_center=(scale_base+scale_min*loc_rate).long()
                    loc_top=loc_center[:,0]-patch_height//2
                    loc_bot=loc_center[:,0]+patch_height//2+patch_height%2
                    loc_lef=loc_center[:,1]-patch_width//2
                    loc_rig=loc_center[:,1]+patch_width//2+patch_width%2
                    loc_tl=torch.stack((loc_top, loc_lef), dim=1)
                    loc_br=torch.stack((loc_bot, loc_rig), dim=1)

                    # For boundary conditions
                    loc_below=loc_tl.detach().clone() # too low
                    loc_below[loc_below>0]=0
                    loc_br-=loc_below
                    loc_tl-=loc_below
                    loc_over=loc_br-scale.long() # too high
                    loc_over[loc_over<0]=0
                    loc_tl-=loc_over
                    loc_br-=loc_over
                    loc_tl[loc_tl<0]=0 # patch too large

                    input_loc_list.append(torch.cat((loc_tl, loc_br), dim=1))

            input_loc_tensor=torch.stack(input_loc_list, dim=1) # (7,6,4)
            return input_loc_tensor


class Patch5Model(nn.Module):
    def __init__(self, num_classes=1, img_width=224, img_height=224, LFS_window_size=10, LFS_M = 6):
        super(Patch5Model, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.model, self.preprocess = clip.load('ViT-L/14', device="cpu")
        self.COOI=COOI()
        self.mha_list=nn.Sequential(
                        SA_layer(256, 8),
                        SA_layer(256, 8),
                        SA_layer(256, 8)
                      )
        self.fc1 = nn.Linear(2048, 256)  # 尝试256->128
        self.fc4 = nn.Linear(768, 256)
        self.ac = nn.ReLU()
        self.fc = nn.Linear(256, 1)
        self.TextureEnhancer = resnet.resnet18(pretrained=True)
        self.weight_fc = nn.Linear(256, 18, bias=False)


        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self._LFS_M = LFS_M
        # init branches
        self.FAD_head = FAD_Head(img_size)
        self.init_xcep_FAD()
        self.LFS_head = LFS_Head(img_size, LFS_window_size, LFS_M)
        self.init_xcep_LFS()
        # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 256)
        self.dp = nn.Dropout(p=0.2)


    def init_xcep_FAD(self):
        self.FAD_xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.FAD_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.FAD_xcep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            self.FAD_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / 4.0

    def init_xcep_LFS(self):
        self.LFS_xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.LFS_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)

    def _norm_fea(self, fea):
        f = self.relu(fea)
        f = F.adaptive_avg_pool2d(f, (1,1))
        f = f.view(f.size(0), -1)
        return f

    def forward(self, input_img, cropped_img, scale):
        x = cropped_img
        batch_size, p, _, _ = x.shape  # [batch_size, 3, 224, 224]
        # TextureGramMatrix Module
        enhanced_texture, _ = self.TextureEnhancer(x)
        s_enhanced_texture = self.ac(enhanced_texture)
        s_enhanced_texture = s_enhanced_texture.view(-1, 1, 256)
        # Global Branch
        features = self.model.encode_image(x)  # feature[batchsize,768]
        s_whole_embedding = self.ac(self.fc4(features))  # 1000->256
        s_whole_embedding = s_whole_embedding.view(-1, 1, 256)
        # Dual-Frequency Module
        fea_FAD = self.FAD_head(x)
        fea_FAD = self.FAD_xcep.features(fea_FAD)
        fea_FAD = self._norm_fea(fea_FAD)
        fea_LFS = self.LFS_head(x)
        fea_LFS = self.LFS_xcep.features(fea_LFS)
        fea_LFS = self._norm_fea(fea_LFS)
        y = torch.cat((fea_FAD, fea_LFS), dim=1)  # y :[batch_size, 4096]
        y = self.dp(y)
        whole_embedding_freq = y
        s_freq_embedding = self.ac(self.fc2(whole_embedding_freq))  # 4096->128
        s_freq_embedding = s_freq_embedding.view(-1, 1, 256)
        # Local Branch
        fm, _, _ = self.resnet(
            x)  # fm[batch_size, 2048, 7, 7], whole_embedding:[batch_size, 2048], fm1[batch_size, 256, 56, 56]
        input_loc = self.COOI.get_coordinates(fm.detach(), scale)

        _, proposal_size, _ = input_loc.size()

        window_imgs = torch.zeros([batch_size, proposal_size, 3, 224, 224]).to(fm.device)  # [N, 4, 3, 224, 224]

        for batch_no in range(batch_size):
            for proposal_no in range(proposal_size):
                t, l, b, r = input_loc[batch_no, proposal_no]
                img_patch=input_img[batch_no][:, t:b, l:r]
                _, patch_height, patch_width=img_patch.size()
                if patch_height==224 and patch_width==224:
                    window_imgs[batch_no, proposal_no]=img_patch
                else:
                    window_imgs[batch_no, proposal_no:proposal_no+1]=F.interpolate(img_patch[None,...], size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)  # [N, 4, 3, 224, 224]

        window_imgs = window_imgs.reshape(batch_size * proposal_size, 3, 224, 224)  # [N*4, 3, 224, 224]
        _, window_embeddings, _ = self.resnet(window_imgs.detach())  # [batchsize*self.proposalN, 2048]
        s_window_embedding = self.ac(self.fc1(window_embeddings))  # [batchsize*self.proposalN, 128]
        s_window_embedding = s_window_embedding.view(-1, proposal_size, 256)
        # Feature Fusion Module
        all_embeddings = torch.cat((s_window_embedding, s_freq_embedding, s_enhanced_texture, s_whole_embedding), 1)
        all_embeddings_fused = self.mha_list(all_embeddings)
        all_embeddings = torch.cat((all_embeddings, all_embeddings_fused), 1)  # [batch_size,18,256]
        all_embeddings = all_embeddings.permute(0, 2, 1)  # 形状变为 [batch_size, feature_dim, num_features]
        weighted_features = torch.matmul(all_embeddings, self.weight_fc.weight.unsqueeze(0))
        fused_feature = torch.sum(weighted_features, dim=2)  # 形状变为 [batch_size, feature_dim]
        all_logits = self.fc(fused_feature)
        return all_logits


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model=Patch5Model()
            if torch.cuda.device_count()>1:
                self.model=nn.DataParallel(self.model)
       

        if not self.isTrain or opt.continue_train:
            #self.model = resnet50(num_classes=1)
            self.model=Patch5Model()
            if torch.cuda.device_count()>1:
                self.model=nn.DataParallel(self.model)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        if len(opt.gpu_ids)==0:
            self.model.to('cpu')
        else:
            self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 2.
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr
                return False
        return True

    def set_input(self, data):
        self.input_img = data[0] # (batch_size, 6, 3, 224, 224)
        self.cropped_img = data[1].to(self.device)
        self.label = data[2].to(self.device).float() #(batch_size)
        self.scale = data[3].to(self.device).float()
        #self.imgname = data[4]

    def forward(self):
        self.output = self.model(self.input_img, self.cropped_img, self.scale)  # output (batch_size, 1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


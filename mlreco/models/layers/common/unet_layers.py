import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.layers.common.blocks import ResNetBlock, CascadeDilationBlock, ASPP
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.activation_normalization_factories import normalizations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration


class UNetEncoder(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    encoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from encoder half
    finalTensor: ME.SparseTensor
        feature tensor at deepest layer
    features_ppn: list of ME.SparseTensor
        list of intermediate tensors (right after encoding block + convolution)
    '''
    def __init__(self, cfg, name='uresnet_encoder'):
        # To allow UResNet to inherit directly from UResNetEncoder
        super(UNetEncoder, self).__init__()
        #torch.nn.Module.__init__(self)
        setup_cnn_configuration(self, cfg, name)

        model_cfg = cfg.get(name, {})
        # UResNet Configurations
        self.reps = model_cfg.get('reps', 2)
        self.depth = model_cfg.get('depth', 5)
        self.num_filters = model_cfg.get('filters', 16)
        self.nPlanes = [i * self.num_filters for i in range(1, self.depth+1)]
        # self.kernel_size = cfg.get('kernel_size', 3)
        # self.downsample = cfg.get(downsample, 2)
        self.input_kernel = model_cfg.get('input_kernel', 3)

        # Initialize Input Layer
        # print(self.num_input)
        # print(self.input_kernel)
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel, stride=1, dimension=self.D,
            bias=self.allow_bias)

        # Initialize Encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension=self.D,
                    activation=self.activation_name,
                    activation_args=self.activation_args,
                    normalization=self.norm,
                    normalization_args=self.norm_args,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm, F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(
                    in_channels=self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size=2, stride=2, dimension=self.D,
                    bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder.

        Parameters
        ----------
        x : MinkowskiEngine SparseTensor

        Returns
        -------
        dict
        '''
        # print('input' , self.input_layer)
        # for name, param in self.input_layer.named_parameters():
        #     print(name, param.shape, param)
        x = self.input_layer(x)
        encoderTensors = [x]
        features_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)
            features_ppn.append(x)

        result = {
            "encoderTensors": encoderTensors,
            "features_ppn": features_ppn,
            "finalTensor": x
        }
        return result


    def forward(self, input):
        # coords = input[:, 0:self.D+1].int()
        # features = input[:, self.D+1:].float()
        #
        # x = ME.SparseTensor(features, coordinates=coords)
        encoderOutput = self.encoder(input)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        # decoderTensors = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': encoderTensors,
            # 'decoderTensors': decoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }
        return res



class UNet3Plus(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    features: last layer from decoder
    '''
    def __init__(self, cfg, name='uresnet'):
        super(UNet3Plus, self).__init__()
        setup_cnn_configuration(self, cfg, name)
        self.encoder = UNetEncoder(cfg, name=name)
        self.decoder = UNet3PlusDecoder(cfg, name=name)

        self.num_filters = self.encoder.num_filters
        self.cat_channels = self.num_filters
        self.upsample_channels = self.cat_channels * self.depth

        # print('Total Number of Trainable Parameters (mink/layers/uresnet) = {}'.format(
        #             sum(p.numel() for p in self.parameters() if p.requires_grad)))

    def forward(self, input):
        coords = input[:, 0:self.D+1].int()
        features = input[:, self.D+1:].float()

        x = ME.SparseTensor(features, coordinates=coords)
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors = self.decoder(encoderTensors[1:], finalTensor)

        res = {
            'encoderTensors': encoderTensors,
            'decoderTensors': decoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }
        return res

class UNet3PlusDecoder(nn.Module):
    def __init__(self, cfg, name):
        super(UNet3PlusDecoder, self).__init__()
        setup_cnn_configuration(self, cfg, name)

        self.model_config = cfg.get(name, {})
        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.depth = self.model_config.get('depth', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.depth+1)]

        # Initialize Decoder
        self.feature_agg = FeatureAggregationBlock(cfg,name=name).cuda()
        self.decoding_block = []

        for _ in range(self.depth-2, -1, -1):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(self.depth*5,
                                     self.depth*5,
                                     dimension=self.D,
                                     activation=self.activation_name,
                                     activation_args=self.activation_args,
                                     normalization=self.norm,
                                     normalization_args=self.norm_args,
                                     bias=self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)

    def decoder(self, encoderTensors, final):
        '''
        Vanilla UResNet Decoder

        Parameters
        ----------
        encoderTensors : list of SparseTensor
            output of encoder.

        Returns
        -------
        decoderTensors : list of SparseTensor
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []

        for _,block in enumerate(self.decoding_block):
            x = self.feature_agg(encoderTensors,decoderTensors,final)
            x = block(x)
            decoderTensors.append(x)
        return decoderTensors

    def forward(self, encoderTensors, final):
        return self.decoder(encoderTensors, final)

# class FeatureAggregationBlock(nn.Module):
#     def __init__(self, cfg, name):
#         super(FeatureAggregationBlock, self).__init__()
#         setup_cnn_configuration(self, cfg, name)
#         self.model_config = cfg.get(name, {})
#         self.depth = self.model_config.get('depth', 5)
#         self.num_filters = self.model_config.get('filters', 16)
#         self.nPlanes = [i*self.num_filters for i in range(1, self.depth+1)]
#         self.Conv = ME.MinkowskiConvolution
#         self.Up = ME.MinkowskiPoolingTranspose
#         self.Down = ME.MinkowskiMaxPooling
#         self.ReLU = ME.MinkowskiReLU
#         self.BN = ME.MinkowskiBatchNorm
#         # self.activation = activations_construct(self.activation_name, **self.activation_args)

#     def forward(self,encoder_tensors, decoder_tensors, final):
#         '''
#         INPUT

#         '''
#         agg = []
#         decoder_id = 1+len(decoder_tensors)         # numerating begins with bottelneck to upper layer
#         encoder_id = self.depth - decoder_id -1  # coresponding encoder index
#         for i in range(0, encoder_id):            # iterating over all upper encoders to pool these layers
#             tensor = encoder_tensors[i]
#             scale_factor = 2**(encoder_id-i)
#             tensor = self.Down(scale_factor,scale_factor,dimension=self.D).cuda()(tensor)
#             tensor = self.Conv(self.nPlanes[i], self.num_filters, 3, dimension=self.D, bias=self.allow_bias).cuda()(tensor)
#             agg.insert(0,tensor)

#         # add features from corresponding encoder
#         tensor = encoder_tensors[encoder_id]
#         agg.insert(0,self.Conv(self.nPlanes[encoder_id],
#                                self.num_filters,
#                                3,dimension=self.D, 
#                                bias=self.allow_bias).cuda()(tensor))

#         # upsample bottom layer and add to stack of feature maps
#         scale_factor = 2**decoder_id
#         tensor = self.Up(scale_factor, scale_factor, dimension=self.D)(final)
#         tensor = self.Conv(self.nPlanes[-1], self.num_filters, 3, dimension=self.D, bias=self.allow_bias).cuda()(tensor)
#         agg.insert(0,tensor)
#         for i,tensor in enumerate(decoder_tensors):
#             scale_factor = 2**(decoder_id-1-i)
#             tensor = self.Up(scale_factor, scale_factor, dimension=self.D).cuda()(tensor)
#             tensor = self.Conv(self.depth*5, self.num_filters, kernel_size=3, dimension=self.D, bias=self.allow_bias).cuda()(tensor)
#             agg.insert(i+1,tensor)

    
#         agg = ME.cat(agg)
#         agg = self.Conv(agg.shape[1], self.depth*5, 3, dimension=self.D, bias=self.allow_bias).cuda()(agg)

#         # agg = self.BN(agg.shape[1])
#         # agg = self.ReLU()(agg)
#         return agg

        
class FeatureAggregationBlock(nn.Module):
    def __init__(self, cfg, name):
        super(FeatureAggregationBlock, self).__init__()
        setup_cnn_configuration(self, cfg, name)
        self.model_config = cfg.get(name, {})
        self.depth = self.model_config.get('depth', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.cat_channels = self.num_filters
        self.upsample_channels = self.cat_channels * self.depth
        self.nPlanes = [i*self.num_filters for i in range(1, self.depth+1)]
        self.Conv = ME.MinkowskiConvolution
        self.Up = ME.MinkowskiPoolingTranspose
        self.Down = ME.MinkowskiMaxPooling
        self.ReLU = ME.MinkowskiReLU
        self.BN = ME.MinkowskiBatchNorm
        self.encod_conv = []
        for i, n in enumerate(self.nPlanes):
            m = nn.Sequential(
                self.Conv(n, self.cat_channels, 3, dimension=self.D, bias=self.allow_bias),
                self.BN(self.cat_channels),
                self.ReLU()
            )
            self.encod_conv.append(m)
        self.encod_conv = nn.Sequential(*self.encod_conv)
        self.decod_conv = nn.Sequential(
            self.Conv(self.upsample_channels, self.cat_channels, kernel_size=3, dimension=self.D, bias=self.allow_bias),
            self.BN(self.cat_channels),
            self.ReLU()
        )
        self.Conv_out = nn.Sequental(
            self.Conv(self.upsample_channels, self.upsample_channels, 3, dimension=self.D, bias=self.allow_bias),
            self.BN(self.upsample_channels),
            self.ReLU()
        )
        # self.activation = activations_construct(self.activation_name, **self.activation_args)

    def forward(self,encoder_tensors, decoder_tensors, final):
        '''
        INPUT

        '''
        agg = []
        decoder_id = 1+len(decoder_tensors)         # numerating begins with bottelneck to upper layer
        encoder_id = self.depth - decoder_id -1  # coresponding encoder index
        for i in range(0, encoder_id):            # iterating over all upper encoders to pool these layers
            tensor = encoder_tensors[i]
            scale_factor = 2**(encoder_id-i)
            tensor = self.Down(scale_factor,scale_factor,dimension=self.D)(tensor)
            tensor = self.encod_conv[i](tensor)
            agg.insert(0,tensor)

        # add features from corresponding encoder
        tensor = encoder_tensors[encoder_id]
        agg.insert(0,self.encod_conv[encoder_id](tensor))

        # upsample bottom layer and add to stack of feature maps
        scale_factor = 2**decoder_id
        tensor = self.Up(scale_factor, scale_factor, dimension=self.D)(final)
        tensor = self.encod_conv[-1](tensor)
        agg.insert(0,tensor)
        for i,tensor in enumerate(decoder_tensors):
            scale_factor = 2**(decoder_id-1-i)
            tensor = self.Up(scale_factor, scale_factor, dimension=self.D)(tensor)
            tensor = self.decod_conv(tensor)
            agg.insert(i+1,tensor)

    
        agg = ME.cat(agg)
        agg = self.Conv_out(agg)

        # agg = self.BN(agg.shape[1])
        # agg = self.ReLU()(agg)
        return agg    

            

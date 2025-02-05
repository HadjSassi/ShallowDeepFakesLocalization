#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : functional.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.
# EDITION -----------------------------------------------
# Edited By : Hadj Sassi Mahdi
# Edition Date : 02/12/2023

import torch
import torch.autograd as ag

__all__ = ['prroi_pool2d']
prroiPooling = None
def _import_prroi_pooling():
    global prroiPooling

    if prroiPooling is None:
        try:
            from os.path import join as pjoin, dirname
            from torch.utils.cpp_extension import load as load_extension
            root_dir = pjoin(dirname(__file__), 'src')

            prroiPooling = load_extension(
                '_prroi_pooling',
                [pjoin(root_dir, 'prroi_pooling_gpu.c'), pjoin(root_dir, 'prroi_pooling_gpu_impl.cu')],
                verbose=True
            )
        except ImportError:
            raise ImportError('Can not compile Precise RoI Pooling library.')

    return prroiPooling


class PrRoIPool2DFunction(ag.Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        pooling = _import_prroi_pooling()

        assert 'FloatTensor' in features.type() and 'FloatTensor' in rois.type(), \
                'Precise RoI Pooling only takes float input, got {} for features and {} for rois.'.format(features.type(), rois.type())

        pooled_height = int(pooled_height)
        pooled_width = int(pooled_width)
        spatial_scale = float(spatial_scale)

        features = features.contiguous()
        rois = rois.contiguous()
        params = (pooled_height, pooled_width, spatial_scale)

        if features.is_cuda:
            output = pooling.prroi_pooling_forward_cuda(features, rois, *params)
            ctx.params = params
            # everything here is contiguous.
            ctx.save_for_backward(features, rois, output)
        else:
            raise NotImplementedError('Precise RoI Pooling only supports GPU (cuda) implememtations.')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        pooling = _import_prroi_pooling()

        features, rois, output = ctx.saved_tensors
        grad_input = grad_coor = None

        if features.requires_grad:
            grad_output = grad_output.contiguous()
            grad_input = pooling.prroi_pooling_backward_cuda(features, rois, output, grad_output, *ctx.params)
        if rois.requires_grad:
            grad_output = grad_output.contiguous()
            grad_coor = pooling.prroi_pooling_coor_backward_cuda(features, rois, output, grad_output, *ctx.params)

        return grad_input, grad_coor, None, None, None

prroi_pool2d = PrRoIPool2DFunction.apply
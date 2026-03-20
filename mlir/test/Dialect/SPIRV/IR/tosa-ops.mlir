// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @argmax_int(%arg0: !spirv.arm.tensor<3x28x17x17xi8>) -> (!spirv.arm.tensor<3x28x17xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  %2 = spirv.Tosa.ArgMax axis = 3, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<3x28x17x17xi8> -> !spirv.arm.tensor<3x28x17xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x28x17xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x28x17xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @argmax_fp(%arg0: !spirv.arm.tensor<2x2x7x14xf32>) -> (!spirv.arm.tensor<2x2x14xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
  %2 = spirv.Tosa.ArgMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x2x7x14xf32> -> !spirv.arm.tensor<2x2x14xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x2x14xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x2x14xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @avgpool2d_int(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32768x1xi8>) {
  %4 = spirv.Constant dense<125> : !spirv.arm.tensor<1xi8>
  %5 = spirv.Constant dense<-90> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
  %6 = spirv.Tosa.AvgPool2D kernel = [3, 3], stride = [1, 2], pad = [0, 1, 0, 0], acc_type = <INT32>, %arg0, %4, %5 : !spirv.arm.tensor<1x3x65537x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x32768x1xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x32768x1xi8>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x32768x1xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @avgpool2d_fp(%arg0: !spirv.arm.tensor<1x2x65533x2xf32>) -> (!spirv.arm.tensor<1x2x65532x2xf32>) {
  %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // CHECK: {{%.*}} = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [1, 1], pad = [1, 0, 0, 0], acc_type = <FP32>, %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
  %6 = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [1, 1], pad = [1, 0, 0, 0], acc_type = <FP32>, %arg0, %4, %5 : !spirv.arm.tensor<1x2x65533x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x2x65532x2xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x65532x2xf32>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x2x65532x2xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv2d_int(%arg0: !spirv.arm.tensor<1x65535x3x1xi8>, %arg1: !spirv.arm.tensor<7x1x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x65536x2x7xi32>) {
  %5 = spirv.Constant dense<35> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<57> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  %7 = spirv.Tosa.Conv2D pad = [1, 0, 0, 0], stride = [1, 2], dilation = [7, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65535x3x1xi8>, !spirv.arm.tensor<7x1x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x65536x2x7xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65536x2x7xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65536x2x7xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv2d_fp(%arg0: !spirv.arm.tensor<1x34x18x27xf16>, %arg1: !spirv.arm.tensor<11x1x1x27xf16>, %arg2: !spirv.arm.tensor<11xf16>) -> (!spirv.arm.tensor<1x34x18x11xf16>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // CHECK: {{%.*}} = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  %7 = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x34x18x27xf16>, !spirv.arm.tensor<11x1x1x27xf16>, !spirv.arm.tensor<11xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<1x34x18x11xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x34x18x11xf16>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x34x18x11xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv3d_int(%arg0: !spirv.arm.tensor<1x9x21x14x1xi8>, %arg1: !spirv.arm.tensor<2x1x2x1x1xi8>, %arg2: !spirv.arm.tensor<1xi32>) -> (!spirv.arm.tensor<1x9x20x14x2xi32>) {
  %5 = spirv.Constant dense<123> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<121> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
  %7 = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x9x21x14x1xi8>, !spirv.arm.tensor<2x1x2x1x1xi8>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x9x20x14x2xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x9x20x14x2xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x9x20x14x2xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @conv3d_fp(%arg0: !spirv.arm.tensor<1x2x65539x1x2xf32>, %arg1: !spirv.arm.tensor<1x1x1x1x2xf32>, %arg2: !spirv.arm.tensor<1xf32>) -> (!spirv.arm.tensor<1x3x65540x2x1xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // CHECK: {{%.*}} = spirv.Tosa.Conv3D pad = [0, 1, 1, 0, 0, 1], stride = [1, 1, 1], dilation = [1, 1, 7], acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
  %7 = spirv.Tosa.Conv3D pad = [0, 1, 1, 0, 0, 1], stride = [1, 1, 1], dilation = [1, 1, 7], acc_type = <FP32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x2x65539x1x2xf32>, !spirv.arm.tensor<1x1x1x1x2xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x3x65540x2x1xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x3x65540x2x1xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x3x65540x2x1xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @depthwiseconv2d_int(%arg0: !spirv.arm.tensor<1x4x65537x1xi8>, %arg1: !spirv.arm.tensor<1x3x1x4xi8>, %arg2: !spirv.arm.tensor<4xi32>) -> (!spirv.arm.tensor<1x4x32762x4xi32>) {
  %5 = spirv.Constant dense<58> : !spirv.arm.tensor<1xi8>
  %6 = spirv.Constant dense<-106> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 2], dilation = [7, 7], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 2], dilation = [7, 7], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x4x65537x1xi8>, !spirv.arm.tensor<1x3x1x4xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x32762x4xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x4x32762x4xi32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x4x32762x4xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @depthwiseconv2d_fp(%arg0: !spirv.arm.tensor<1x65540x1x3xf32>, %arg1: !spirv.arm.tensor<1x1x3x1xf32>, %arg2: !spirv.arm.tensor<1xf32>) -> (!spirv.arm.tensor<1x65541x2x3xf32>) {
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  %6 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf32>
  // CHECK: {{%.*}} = spirv.Tosa.DepthwiseConv2D pad = [0, 1, 1, 1], stride = [1, 2], dilation = [1, 7], acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
  %7 = spirv.Tosa.DepthwiseConv2D pad = [0, 1, 1, 1], stride = [1, 2], dilation = [1, 7], acc_type = <FP32>, local_bound = true, %arg0, %arg1, %arg2, %5, %6 : !spirv.arm.tensor<1x65540x1x3xf32>, !spirv.arm.tensor<1x1x3x1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32>, !spirv.arm.tensor<1xf32> -> !spirv.arm.tensor<1x65541x2x3xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65541x2x3xf32>
  spirv.ARM.GraphOutputs %7 : !spirv.arm.tensor<1x65541x2x3xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.FFT2D - EXT-FFT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @fft2d_fft(%arg0: !spirv.arm.tensor<1x32x32xf32>, %arg1: !spirv.arm.tensor<1x32x32xf32>) -> (!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.FFT2D inverse = true, local_bound = false, %arg0, %arg1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  %out = spirv.Tosa.FFT2D inverse = true, local_bound = false, %arg0, %arg1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  %out0 = spirv.CompositeExtract %out[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] :  !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  %out1 = spirv.CompositeExtract %out[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>
  spirv.ARM.GraphOutputs  %out0, %out1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @matmul_int(%arg0: !spirv.arm.tensor<8x2x3xi8>, %arg1: !spirv.arm.tensor<8x3x8xi8>) -> (!spirv.arm.tensor<8x2x8xi32>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  %1 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.MatMul %arg0, %arg1, {{%.*}}, {{%.*}} : !spirv.arm.tensor<8x2x3xi8>, !spirv.arm.tensor<8x3x8xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<8x2x8xi32>
  %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<8x2x3xi8>, !spirv.arm.tensor<8x3x8xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<8x2x8xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<8x2x8xi32>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<8x2x8xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @matmul_fp(%arg0: !spirv.arm.tensor<15x39x50xf16>, %arg1: !spirv.arm.tensor<15x50x24xf16>) -> (!spirv.arm.tensor<15x39x24xf16>) {
  %0 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %1 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // CHECK: {{%.*}} = spirv.Tosa.MatMul %arg0, %arg1, {{%.*}}, {{%.*}} : !spirv.arm.tensor<15x39x50xf16>, !spirv.arm.tensor<15x50x24xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<15x39x24xf16>
  %2 = spirv.Tosa.MatMul %arg0, %arg1, %0, %1 : !spirv.arm.tensor<15x39x50xf16>, !spirv.arm.tensor<15x50x24xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<15x39x24xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<15x39x24xf16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<15x39x24xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maxpool2d_int(%arg0: !spirv.arm.tensor<1x3x65537x1xi8>) -> (!spirv.arm.tensor<1x2x32769x1xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [1, 2], pad = [1, 0, 0, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
  %4 = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [1, 2], pad = [1, 0, 0, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x3x65537x1xi8> -> !spirv.arm.tensor<1x2x32769x1xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x32769x1xi8>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x2x32769x1xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maxpool2d_fp(%arg0: !spirv.arm.tensor<1x6x65536x1xf32>) -> (!spirv.arm.tensor<1x3x32769x1xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [2, 2], pad = [1, 0, 1, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
  %4 = spirv.Tosa.MaxPool2D kernel = [3, 2], stride = [2, 2], pad = [1, 0, 1, 1], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x6x65536x1xf32> -> !spirv.arm.tensor<1x3x32769x1xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x3x32769x1xf32>
  spirv.ARM.GraphOutputs %4 : !spirv.arm.tensor<1x3x32769x1xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.RFFT2D - EXT-FFT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @rfft2d_fft(%arg0: !spirv.arm.tensor<1x32x32xf32>) -> (!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.RFFT2D local_bound = false, %arg0 : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  %out = spirv.Tosa.RFFT2D local_bound = false, %arg0 : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  %out0 = spirv.CompositeExtract %out[0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  // CHECK: {{%.*}} = spirv.CompositeExtract {{%.*}}[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  %out1 = spirv.CompositeExtract %out[1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>
  spirv.ARM.GraphOutputs %out0, %out1 : !spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @transposeconv2d_int(%arg0: !spirv.arm.tensor<1x13x33x3xi16>, %arg1: !spirv.arm.tensor<11x1x3x3xi8>, %arg2: !spirv.arm.tensor<1xi64>) -> (!spirv.arm.tensor<1x13x35x11xi64>) {
  %4 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  %5 = spirv.Constant dense<88> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
  %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT48>, local_bound = false, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<1x13x33x3xi16>, !spirv.arm.tensor<11x1x3x3xi8>, !spirv.arm.tensor<1xi64>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x13x35x11xi64>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x13x35x11xi64>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<1x13x35x11xi64>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @transposeconv2d_fp(%arg0: !spirv.arm.tensor<10x24x9x13xf16>, %arg1: !spirv.arm.tensor<14x1x1x13xf16>, %arg2: !spirv.arm.tensor<14xf16>) -> (!spirv.arm.tensor<10x25x65x14xf16>) {
  %4 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %5 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // CHECK: {{%.*}} = spirv.Tosa.TransposeConv2D out_pad = [0, 1, 0, 0], stride = [1, 8], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, {{%.*}}, {{%.*}} : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
  %6 = spirv.Tosa.TransposeConv2D out_pad = [0, 1, 0, 0], stride = [1, 8], acc_type = <FP16>, local_bound = true, %arg0, %arg1, %arg2, %4, %5 : !spirv.arm.tensor<10x24x9x13xf16>, !spirv.arm.tensor<14x1x1x13xf16>, !spirv.arm.tensor<14xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<10x25x65x14xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x25x65x14xf16>
  spirv.ARM.GraphOutputs %6 : !spirv.arm.tensor<10x25x65x14xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @clamp_int(%arg0: !spirv.arm.tensor<27x44x55xi8>) -> (!spirv.arm.tensor<27x44x55xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.Clamp min_val = -102 : i8, max_val = -100 : i8, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
  %3 = spirv.Tosa.Clamp min_val = -102 : i8, max_val = -100 : i8, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x44x55xi8> -> !spirv.arm.tensor<27x44x55xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<27x44x55xi8>
  spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<27x44x55xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @clamp_fp(%arg0: !spirv.arm.tensor<18x5x17x6xf32>) -> (!spirv.arm.tensor<18x5x17x6xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Clamp min_val = -1.19339396E+38 : f32, max_val = 2.38255944E+38 : f32, nan_mode = <Ignore>, %arg0 : !spirv.arm.tensor<18x5x17x6xf32> -> !spirv.arm.tensor<18x5x17x6xf32>
  %3 = spirv.Tosa.Clamp min_val = -1.19339396E+38 : f32, max_val = 2.38255944E+38 : f32, nan_mode = <Ignore>, %arg0 : !spirv.arm.tensor<18x5x17x6xf32> -> !spirv.arm.tensor<18x5x17x6xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<18x5x17x6xf32>
  spirv.ARM.GraphOutputs %3 : !spirv.arm.tensor<18x5x17x6xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Erf - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @erf_fp(%arg0: !spirv.arm.tensor<47x38x51xf32>) -> (!spirv.arm.tensor<47x38x51xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
  %0 = spirv.Tosa.Erf %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<47x38x51xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<47x38x51xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sigmoid - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sigmoid_fp(%arg0: !spirv.arm.tensor<28x43x45xf32>) -> (!spirv.arm.tensor<28x43x45xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
  %0 = spirv.Tosa.Sigmoid %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<28x43x45xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<28x43x45xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tanh - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @tanh_fp(%arg0: !spirv.arm.tensor<46x50x36xf16>) -> (!spirv.arm.tensor<46x50x36xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
  %0 = spirv.Tosa.Tanh %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<46x50x36xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x50x36xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @add_int(%arg0: !spirv.arm.tensor<4x7x3x10xi32>, %arg1: !spirv.arm.tensor<4x7x3x1xi32>) -> (!spirv.arm.tensor<4x7x3x10xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x7x3x10xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x7x3x10xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @add_fp(%arg0: !spirv.arm.tensor<26x37x18xf16>, %arg1: !spirv.arm.tensor<1x37x18xf16>) -> (!spirv.arm.tensor<26x37x18xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<26x37x18xf16>, !spirv.arm.tensor<1x37x18xf16> -> !spirv.arm.tensor<26x37x18xf16>
  %0 = spirv.Tosa.Add %arg0, %arg1 : !spirv.arm.tensor<26x37x18xf16>, !spirv.arm.tensor<1x37x18xf16> -> !spirv.arm.tensor<26x37x18xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<26x37x18xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<26x37x18xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArithmeticRightShift - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @arithmeticrightshift_int(%arg0: !spirv.arm.tensor<1x47x22xi16>, %arg1: !spirv.arm.tensor<49x47x22xi16>) -> (!spirv.arm.tensor<49x47x22xi16>) {
  // CHECK: {{%.*}} = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<1x47x22xi16>, !spirv.arm.tensor<49x47x22xi16> -> !spirv.arm.tensor<49x47x22xi16>
  %1 = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<1x47x22xi16>, !spirv.arm.tensor<49x47x22xi16> -> !spirv.arm.tensor<49x47x22xi16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<49x47x22xi16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<49x47x22xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseAnd - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwiseand_int(%arg0: !spirv.arm.tensor<4x1x7x12xi16>, %arg1: !spirv.arm.tensor<4x13x7x12xi16>) -> (!spirv.arm.tensor<4x13x7x12xi16>) {
  // CHECK: {{%.*}} = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
  %0 = spirv.Tosa.BitwiseAnd %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x13x7x12xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x13x7x12xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseOr - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwiseor_int(%arg0: !spirv.arm.tensor<11x30x23xi32>, %arg1: !spirv.arm.tensor<1x30x23xi32>) -> (!spirv.arm.tensor<11x30x23xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
  %0 = spirv.Tosa.BitwiseOr %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x30x23xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x30x23xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseXor - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwisexor_int(%arg0: !spirv.arm.tensor<4x8x13x9xi16>, %arg1: !spirv.arm.tensor<4x8x1x9xi16>) -> (!spirv.arm.tensor<4x8x13x9xi16>) {
  // CHECK: {{%.*}} = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
  %0 = spirv.Tosa.BitwiseXor %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x8x13x9xi16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x8x13x9xi16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.IntDiv - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @intdiv_any(%arg0: !spirv.arm.tensor<1x65533x1xi32>, %arg1: !spirv.arm.tensor<2x65533x1xi32>) -> (!spirv.arm.tensor<2x65533x1xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
  %0 = spirv.Tosa.IntDiv %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x65533x1xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x65533x1xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalAnd - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicaland_any(%arg0: !spirv.arm.tensor<2x1x7x11xi1>, %arg1: !spirv.arm.tensor<2x4x7x11xi1>) -> (!spirv.arm.tensor<2x4x7x11xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
  %0 = spirv.Tosa.LogicalAnd %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x4x7x11xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<2x4x7x11xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalLeftShift - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalleftshift_any(%arg0: !spirv.arm.tensor<7x1x11x4xi8>, %arg1: !spirv.arm.tensor<7x8x11x4xi8>) -> (!spirv.arm.tensor<7x8x11x4xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
  %0 = spirv.Tosa.LogicalLeftShift %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<7x8x11x4xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<7x8x11x4xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalRightShift - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalrightshift_any(%arg0: !spirv.arm.tensor<6x13x1x19xi8>, %arg1: !spirv.arm.tensor<6x13x6x19xi8>) -> (!spirv.arm.tensor<6x13x6x19xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
  %0 = spirv.Tosa.LogicalRightShift %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x13x6x19xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x13x6x19xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalOr - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalor_any(%arg0: !spirv.arm.tensor<3x6x12x5xi1>, %arg1: !spirv.arm.tensor<3x6x1x5xi1>) -> (!spirv.arm.tensor<3x6x12x5xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
  %0 = spirv.Tosa.LogicalOr %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x6x12x5xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x12x5xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalXor - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalxor_any(%arg0: !spirv.arm.tensor<11x4x9x12xi1>, %arg1: !spirv.arm.tensor<11x4x9x1xi1>) -> (!spirv.arm.tensor<11x4x9x12xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
  %0 = spirv.Tosa.LogicalXor %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x4x9x12xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x4x9x12xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maximum_int(%arg0: !spirv.arm.tensor<1x2x65533x1xi32>, %arg1: !spirv.arm.tensor<1x2x65533x2xi32>) -> (!spirv.arm.tensor<1x2x65533x2xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
  %1 = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x2x65533x2xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<1x2x65533x2xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @maximum_fp(%arg0: !spirv.arm.tensor<1x12x14x7xf16>, %arg1: !spirv.arm.tensor<11x12x14x7xf16>) -> (!spirv.arm.tensor<11x12x14x7xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Maximum nan_mode = <Ignore>, %arg0, %arg1 : !spirv.arm.tensor<1x12x14x7xf16>, !spirv.arm.tensor<11x12x14x7xf16> -> !spirv.arm.tensor<11x12x14x7xf16>
  %1 = spirv.Tosa.Maximum nan_mode = <Ignore>, %arg0, %arg1 : !spirv.arm.tensor<1x12x14x7xf16>, !spirv.arm.tensor<11x12x14x7xf16> -> !spirv.arm.tensor<11x12x14x7xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x12x14x7xf16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<11x12x14x7xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @minimum_int(%arg0: !spirv.arm.tensor<15x2x10x11xi32>, %arg1: !spirv.arm.tensor<15x1x10x11xi32>) -> (!spirv.arm.tensor<15x2x10x11xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
  %1 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<15x2x10x11xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<15x2x10x11xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @minimum_fp(%arg0: !spirv.arm.tensor<1x65531x2x1xf32>, %arg1: !spirv.arm.tensor<1x1x2x1xf32>) -> (!spirv.arm.tensor<1x65531x2x1xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x65531x2x1xf32>, !spirv.arm.tensor<1x1x2x1xf32> -> !spirv.arm.tensor<1x65531x2x1xf32>
  %1 = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x65531x2x1xf32>, !spirv.arm.tensor<1x1x2x1xf32> -> !spirv.arm.tensor<1x65531x2x1xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<1x65531x2x1xf32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<1x65531x2x1xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @mul_int(%arg0: !spirv.arm.tensor<34x21x39xi32>, %arg1: !spirv.arm.tensor<34x21x1xi32>) -> (!spirv.arm.tensor<34x21x39xi32>) {
  %0 = spirv.Constant dense<31> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Mul %arg0, %arg1, {{%.*}} : !spirv.arm.tensor<34x21x39xi32>, !spirv.arm.tensor<34x21x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<34x21x39xi32>, !spirv.arm.tensor<34x21x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<34x21x39xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<34x21x39xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<34x21x39xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @mul_fp(%arg0: !spirv.arm.tensor<57x1x55xf16>, %arg1: !spirv.arm.tensor<57x37x55xf16>) -> (!spirv.arm.tensor<57x37x55xf16>) {
  %0 = spirv.Constant dense<0> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Mul %arg0, %arg1, {{%.*}} : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
  %1 = spirv.Tosa.Mul %arg0, %arg1, %0 : !spirv.arm.tensor<57x1x55xf16>, !spirv.arm.tensor<57x37x55xf16>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<57x37x55xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<57x37x55xf16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<57x37x55xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pow - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @pow_fp(%arg0: !spirv.arm.tensor<1x52x53xf16>, %arg1: !spirv.arm.tensor<44x52x53xf16>) -> (!spirv.arm.tensor<44x52x53xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
  %0 = spirv.Tosa.Pow %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<44x52x53xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x52x53xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sub_int(%arg0: !spirv.arm.tensor<6x10x6x6xi32>, %arg1: !spirv.arm.tensor<1x10x6x6xi32>) -> (!spirv.arm.tensor<6x10x6x6xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x10x6x6xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x6x6xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sub_fp(%arg0: !spirv.arm.tensor<1x10x13x12xf16>, %arg1: !spirv.arm.tensor<6x10x13x12xf16>) -> (!spirv.arm.tensor<6x10x13x12xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<1x10x13x12xf16>, !spirv.arm.tensor<6x10x13x12xf16> -> !spirv.arm.tensor<6x10x13x12xf16>
  %0 = spirv.Tosa.Sub %arg0, %arg1 : !spirv.arm.tensor<1x10x13x12xf16>, !spirv.arm.tensor<6x10x13x12xf16> -> !spirv.arm.tensor<6x10x13x12xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x10x13x12xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x10x13x12xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Table - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @table_int(%arg0: !spirv.arm.tensor<3x2x15x7xi8>) -> (!spirv.arm.tensor<3x2x15x7xi8>) {
  %0 = spirv.ARM.GraphConstant {graph_constant_id = 0 : i32} : !spirv.arm.tensor<256xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Table %arg0, {{%.*}} : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
  %1 = spirv.Tosa.Table %arg0, %0 : !spirv.arm.tensor<3x2x15x7xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2x15x7xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x2x15x7xi8>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<3x2x15x7xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @abs_int(%arg0: !spirv.arm.tensor<5x1x4x4xi32>) -> (!spirv.arm.tensor<5x1x4x4xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
  %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<5x1x4x4xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<5x1x4x4xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @abs_fp(%arg0: !spirv.arm.tensor<3x6x14x8xf16>) -> (!spirv.arm.tensor<3x6x14x8xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf16>
  %0 = spirv.Tosa.Abs %arg0 : !spirv.arm.tensor<3x6x14x8xf16> -> !spirv.arm.tensor<3x6x14x8xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x6x14x8xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x6x14x8xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseNot - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @bitwisenot_int(%arg0: !spirv.arm.tensor<12x56x50xi32>) -> (!spirv.arm.tensor<12x56x50xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
  %0 = spirv.Tosa.BitwiseNot %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<12x56x50xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<12x56x50xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Ceil - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @ceil_fp(%arg0: !spirv.arm.tensor<46x55x53xf16>) -> (!spirv.arm.tensor<46x55x53xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
  %0 = spirv.Tosa.Ceil %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<46x55x53xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<46x55x53xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clz - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @clz_int(%arg0: !spirv.arm.tensor<14x10x7x5xi32>) -> (!spirv.arm.tensor<14x10x7x5xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
  %0 = spirv.Tosa.Clz %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<14x10x7x5xi32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<14x10x7x5xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cos - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @cos_fp(%arg0: !spirv.arm.tensor<44x49x51xf32>) -> (!spirv.arm.tensor<44x49x51xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
  %0 = spirv.Tosa.Cos %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<44x49x51xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<44x49x51xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Exp - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @exp_fp(%arg0: !spirv.arm.tensor<37x53x47xf32>) -> (!spirv.arm.tensor<37x53x47xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
  %0 = spirv.Tosa.Exp %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<37x53x47xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<37x53x47xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Floor - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @floor_fp(%arg0: !spirv.arm.tensor<40x52x42xf32>) -> (!spirv.arm.tensor<40x52x42xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
  %0 = spirv.Tosa.Floor %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<40x52x42xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x52x42xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Log - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @log_fp(%arg0: !spirv.arm.tensor<45x43x36xf16>) -> (!spirv.arm.tensor<45x43x36xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
  %0 = spirv.Tosa.Log %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<45x43x36xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<45x43x36xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalNot - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @logicalnot_any(%arg0: !spirv.arm.tensor<54x26x10xi1>) -> (!spirv.arm.tensor<54x26x10xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
  %0 = spirv.Tosa.LogicalNot %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<54x26x10xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<54x26x10xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @negate_int(%arg0: !spirv.arm.tensor<3x1x65540x1xi8>) -> (!spirv.arm.tensor<3x1x65540x1xi8>) {
  %0 = spirv.Constant dense<111> : !spirv.arm.tensor<1xi8>
  %1 = spirv.Constant dense<-32> : !spirv.arm.tensor<1xi8>
  // CHECK: {{%.*}} = spirv.Tosa.Negate %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<3x1x65540x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<3x1x65540x1xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x1x65540x1xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<3x1x65540x1xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @negate_fp(%arg0: !spirv.arm.tensor<2x7x15x13xf16>) -> (!spirv.arm.tensor<2x7x15x13xf16>) {
  %0 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  %1 = spirv.Constant dense<0.000000e+00> : !spirv.arm.tensor<1xf16>
  // CHECK: {{%.*}} = spirv.Tosa.Negate %arg0, {{%.*}}, {{%.*}} : !spirv.arm.tensor<2x7x15x13xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<2x7x15x13xf16>
  %2 = spirv.Tosa.Negate %arg0, %0, %1 : !spirv.arm.tensor<2x7x15x13xf16>, !spirv.arm.tensor<1xf16>, !spirv.arm.tensor<1xf16> -> !spirv.arm.tensor<2x7x15x13xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x7x15x13xf16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x7x15x13xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reciprocal - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reciprocal_fp(%arg0: !spirv.arm.tensor<38x47x44xf32>) -> (!spirv.arm.tensor<38x47x44xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
  %0 = spirv.Tosa.Reciprocal %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<38x47x44xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<38x47x44xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rsqrt - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @rsqrt_fp(%arg0: !spirv.arm.tensor<40x57x56xf32>) -> (!spirv.arm.tensor<40x57x56xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
  %0 = spirv.Tosa.Rsqrt %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<40x57x56xf32>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<40x57x56xf32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sin - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @sin_fp(%arg0: !spirv.arm.tensor<49x38x58xf16>) -> (!spirv.arm.tensor<49x38x58xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
  %0 = spirv.Tosa.Sin %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<49x38x58xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<49x38x58xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @select_int(%arg0: !spirv.arm.tensor<4x1x4x5xi1>, %arg1: !spirv.arm.tensor<4x6x4x5xi8>, %arg2: !spirv.arm.tensor<4x6x4x5xi8>) -> (!spirv.arm.tensor<4x6x4x5xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<4x1x4x5xi1>, !spirv.arm.tensor<4x6x4x5xi8>, !spirv.arm.tensor<4x6x4x5xi8> -> !spirv.arm.tensor<4x6x4x5xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<4x6x4x5xi8>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<4x6x4x5xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @select_fp(%arg0: !spirv.arm.tensor<9x2x15x8xi1>, %arg1: !spirv.arm.tensor<9x2x15x8xf16>, %arg2: !spirv.arm.tensor<9x1x15x8xf16>) -> (!spirv.arm.tensor<9x2x15x8xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
  %0 = spirv.Tosa.Select %arg0, %arg1, %arg2 : !spirv.arm.tensor<9x2x15x8xi1>, !spirv.arm.tensor<9x2x15x8xf16>, !spirv.arm.tensor<9x1x15x8xf16> -> !spirv.arm.tensor<9x2x15x8xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<9x2x15x8xf16>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<9x2x15x8xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @equal_int(%arg0: !spirv.arm.tensor<51x28x59xi32>, %arg1: !spirv.arm.tensor<51x1x59xi32>) -> (!spirv.arm.tensor<51x28x59xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
  %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<51x28x59xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<51x28x59xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @equal_fp(%arg0: !spirv.arm.tensor<16x11x5x3xf32>, %arg1: !spirv.arm.tensor<16x1x5x3xf32>) -> (!spirv.arm.tensor<16x11x5x3xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
  %0 = spirv.Tosa.Equal %arg0, %arg1 : !spirv.arm.tensor<16x11x5x3xf32>, !spirv.arm.tensor<16x1x5x3xf32> -> !spirv.arm.tensor<16x11x5x3xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<16x11x5x3xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<16x11x5x3xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greater_int(%arg0: !spirv.arm.tensor<11x10x10x2xi32>, %arg1: !spirv.arm.tensor<11x10x10x1xi32>) -> (!spirv.arm.tensor<11x10x10x2xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
  %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<11x10x10x2xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<11x10x10x2xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greater_fp(%arg0: !spirv.arm.tensor<6x3x12x4xf16>, %arg1: !spirv.arm.tensor<6x3x1x4xf16>) -> (!spirv.arm.tensor<6x3x12x4xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
  %0 = spirv.Tosa.Greater %arg0, %arg1 : !spirv.arm.tensor<6x3x12x4xf16>, !spirv.arm.tensor<6x3x1x4xf16> -> !spirv.arm.tensor<6x3x12x4xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<6x3x12x4xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<6x3x12x4xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greaterequal_int(%arg0: !spirv.arm.tensor<10x17x7x1xi32>, %arg1: !spirv.arm.tensor<10x17x7x16xi32>) -> (!spirv.arm.tensor<10x17x7x16xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
  %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<10x17x7x16xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<10x17x7x16xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @greaterequal_fp(%arg0: !spirv.arm.tensor<3x17x6x3xf32>, %arg1: !spirv.arm.tensor<1x17x6x3xf32>) -> (!spirv.arm.tensor<3x17x6x3xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
  %0 = spirv.Tosa.GreaterEqual %arg0, %arg1 : !spirv.arm.tensor<3x17x6x3xf32>, !spirv.arm.tensor<1x17x6x3xf32> -> !spirv.arm.tensor<3x17x6x3xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<3x17x6x3xi1>
  spirv.ARM.GraphOutputs %0 : !spirv.arm.tensor<3x17x6x3xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAll - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceall_any(%arg0: !spirv.arm.tensor<18x22x23x12xi1>) -> (!spirv.arm.tensor<18x22x1x12xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceAll axis = 2, %arg0 : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x1x12xi1>
  %1 = spirv.Tosa.ReduceAll axis = 2, %arg0 : !spirv.arm.tensor<18x22x23x12xi1> -> !spirv.arm.tensor<18x22x1x12xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<18x22x1x12xi1>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<18x22x1x12xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAny - PRO-INT or PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceany_any(%arg0: !spirv.arm.tensor<25x13x30x8xi1>) -> (!spirv.arm.tensor<25x13x1x8xi1>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceAny axis = 2, %arg0 : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x1x8xi1>
  %1 = spirv.Tosa.ReduceAny axis = 2, %arg0 : !spirv.arm.tensor<25x13x30x8xi1> -> !spirv.arm.tensor<25x13x1x8xi1>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<25x13x1x8xi1>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<25x13x1x8xi1>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemax_int(%arg0: !spirv.arm.tensor<8x30x12x3xi8>) -> (!spirv.arm.tensor<8x30x1x3xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x1x3xi8>
  %2 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<8x30x12x3xi8> -> !spirv.arm.tensor<8x30x1x3xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<8x30x1x3xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<8x30x1x3xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemax_fp(%arg0: !spirv.arm.tensor<16x20x10xf16>) -> (!spirv.arm.tensor<16x20x1xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x1xf16>
  %2 = spirv.Tosa.ReduceMax axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<16x20x10xf16> -> !spirv.arm.tensor<16x20x1xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<16x20x1xf16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<16x20x1xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemin_int(%arg0: !spirv.arm.tensor<2x5x5x1xi8>) -> (!spirv.arm.tensor<2x5x1x1xi8>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1x1xi8>
  %2 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x5x5x1xi8> -> !spirv.arm.tensor<2x5x1x1xi8>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x5x1x1xi8>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<2x5x1x1xi8>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducemin_fp(%arg0: !spirv.arm.tensor<27x10x25x9xf16>) -> (!spirv.arm.tensor<27x10x1x9xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x1x9xf16>
  %2 = spirv.Tosa.ReduceMin axis = 2, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<27x10x25x9xf16> -> !spirv.arm.tensor<27x10x1x9xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<27x10x1x9xf16>
  spirv.ARM.GraphOutputs %2 : !spirv.arm.tensor<27x10x1x9xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceProduct - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reduceproduct_fp(%arg0: !spirv.arm.tensor<2x16x25xf16>) -> (!spirv.arm.tensor<2x16x1xf16>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceProduct axis = 2, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x1xf16>
  %1 = spirv.Tosa.ReduceProduct axis = 2, %arg0 : !spirv.arm.tensor<2x16x25xf16> -> !spirv.arm.tensor<2x16x1xf16>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<2x16x1xf16>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<2x16x1xf16>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum - PRO-INT
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducesum_int(%arg0: !spirv.arm.tensor<20x24x22xi32>) -> (!spirv.arm.tensor<20x1x22xi32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x1x22xi32>
  %1 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<20x24x22xi32> -> !spirv.arm.tensor<20x1x22xi32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<20x1x22xi32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<20x1x22xi32>
}

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum - PRO-FP
//===----------------------------------------------------------------------===//

spirv.ARM.Graph @reducesum_fp(%arg0: !spirv.arm.tensor<32x32x33xf32>) -> (!spirv.arm.tensor<32x1x33xf32>) {
  // CHECK: {{%.*}} = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x1x33xf32>
  %1 = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<32x32x33xf32> -> !spirv.arm.tensor<32x1x33xf32>
  // CHECK: spirv.ARM.GraphOutputs {{%.*}} : !spirv.arm.tensor<32x1x33xf32>
  spirv.ARM.GraphOutputs %1 : !spirv.arm.tensor<32x1x33xf32>
}

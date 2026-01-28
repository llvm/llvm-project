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

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

// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa %s | FileCheck %s

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArgMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @argmax_int
func.func @argmax_int(%arg0: tensor<2x3x4xi8>) -> tensor<2x4xi32> {
  // CHECK: %[[ARGMAX:.*]] = spirv.Tosa.ArgMax axis = 1, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x4xi32>
  %res = tosa.argmax %arg0 {axis = 1 : i32, nan_mode = PROPAGATE} : (tensor<2x3x4xi8>) -> tensor<2x4xi32>
  return %res : tensor<2x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.AvgPool2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @avg_pool2d_int
func.func @avg_pool2d_int(%arg0: tensor<1x4x4x1xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1x2x2x1xi8> {
  // CHECK: %[[AVG_POOL:.*]] = spirv.Tosa.AvgPool2D kernel = [2, 2], stride = [2, 2], pad = [0, 0, 0, 0], acc_type = <INT32>, %arg0, %arg1, %arg2 : !spirv.arm.tensor<1x4x4x1xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x2x1xi8>
  %res = tosa.avg_pool2d %arg0, %arg1, %arg2 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, acc_type = i32} : (tensor<1x4x4x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x1xi8>
  return %res : tensor<1x2x2x1xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @conv2d_int
func.func @conv2d_int(%arg0: tensor<1x4x4x1xi8>, %arg1: tensor<2x3x3x1xi8>, %arg2: tensor<2xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x2x2x2xi32> {
  // CHECK: %[[CONV2D:.*]] = spirv.Tosa.Conv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %arg3, %arg4 : !spirv.arm.tensor<1x4x4x1xi8>, !spirv.arm.tensor<2x3x3x1xi8>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x2x2xi32>
  %res = tosa.conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = i32, local_bound = false} : (tensor<1x4x4x1xi8>, tensor<2x3x3x1xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x2xi32>
  return %res : tensor<1x2x2x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Conv3D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @conv3d_int
func.func @conv3d_int(%arg0: tensor<1x4x4x4x1xi8>, %arg1: tensor<2x2x2x2x1xi8>, %arg2: tensor<2xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x3x3x3x2xi32> {
  // CHECK: %[[CONV3D:.*]] = spirv.Tosa.Conv3D pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1], dilation = [1, 1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %arg3, %arg4 : !spirv.arm.tensor<1x4x4x4x1xi8>, !spirv.arm.tensor<2x2x2x2x1xi8>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x3x3x3x2xi32>
  %res = tosa.conv3d %arg0, %arg1, %arg2, %arg3, %arg4 {pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>, dilation = array<i64: 1, 1, 1>, acc_type = i32, local_bound = false} : (tensor<1x4x4x4x1xi8>, tensor<2x2x2x2x1xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x3x2xi32>
  return %res : tensor<1x3x3x3x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.DepthwiseConv2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @depthwise_conv2d_int
func.func @depthwise_conv2d_int(%arg0: tensor<1x4x4x1xi8>, %arg1: tensor<3x3x1x2xi8>, %arg2: tensor<2xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x2x2x2xi32> {
  // CHECK: %[[DEPTHWISE_CONV2D:.*]] = spirv.Tosa.DepthwiseConv2D pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %arg3, %arg4 : !spirv.arm.tensor<1x4x4x1xi8>, !spirv.arm.tensor<3x3x1x2xi8>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x2x2xi32>
  %res = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>, acc_type = i32, local_bound = false} : (tensor<1x4x4x1xi8>, tensor<3x3x1x2xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x2x2xi32>
  return %res : tensor<1x2x2x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.FFT2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @fft2d_fp
func.func @fft2d_fp(%arg0: tensor<1x32x32xf32>, %arg1: tensor<1x32x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>) {
  // CHECK: %[[FFT:.*]] = spirv.Tosa.FFT2D inverse = true, local_bound = false, %arg0, %arg1 : !spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  // CHECK: %[[REAL:.*]] = spirv.CompositeExtract %[[FFT]][0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  // CHECK: %[[IMAG:.*]] = spirv.CompositeExtract %[[FFT]][1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x32xf32>, !spirv.arm.tensor<1x32x32xf32>)>
  %real, %imag = "tosa.fft2d"(%arg0, %arg1) <{inverse = true, local_bound = false}> : (tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
  return %real, %imag : tensor<1x32x32xf32>, tensor<1x32x32xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MatMul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @matmul_int
func.func @matmul_int(%arg0: tensor<1x2x3xi8>, %arg1: tensor<1x3x4xi8>, %arg2: tensor<1xi8>, %arg3: tensor<1xi8>) -> tensor<1x2x4xi32> {
  // CHECK: %[[MATMUL:.*]] = spirv.Tosa.MatMul  %arg0, %arg1, %arg2, %arg3 : !spirv.arm.tensor<1x2x3xi8>, !spirv.arm.tensor<1x3x4xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x2x4xi32>
  %res = tosa.matmul %arg0, %arg1, %arg2, %arg3 : (tensor<1x2x3xi8>, tensor<1x3x4xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x2x4xi32>
  return %res : tensor<1x2x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.MaxPool2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @max_pool2d_fp
func.func @max_pool2d_fp(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
  // CHECK: %[[MAX_POOL:.*]] = spirv.Tosa.MaxPool2D kernel = [2, 2], stride = [2, 2], pad = [0, 0, 0, 0], nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<1x4x4x1xf32> -> !spirv.arm.tensor<1x2x2x1xf32>
  %res = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, stride = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, nan_mode = PROPAGATE} : (tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32>
  return %res : tensor<1x2x2x1xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.RFFT2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @rfft2d_fp
func.func @rfft2d_fp(%arg0: tensor<1x32x32xf32>) -> (tensor<1x32x17xf32>, tensor<1x32x17xf32>) {
  // CHECK: %[[RFFT:.*]] = spirv.Tosa.RFFT2D local_bound = false, %arg0 : !spirv.arm.tensor<1x32x32xf32> -> !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  // CHECK: %[[REAL:.*]] = spirv.CompositeExtract %[[RFFT]][0 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  // CHECK: %[[IMAG:.*]] = spirv.CompositeExtract %[[RFFT]][1 : i32] : !spirv.struct<(!spirv.arm.tensor<1x32x17xf32>, !spirv.arm.tensor<1x32x17xf32>)>
  %real, %imag = "tosa.rfft2d"(%arg0) <{local_bound = false}> : (tensor<1x32x32xf32>) -> (tensor<1x32x17xf32>, tensor<1x32x17xf32>)
  return %real, %imag : tensor<1x32x17xf32>, tensor<1x32x17xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.TransposeConv2D
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @transpose_conv2d_int
func.func @transpose_conv2d_int(%arg0: tensor<1x2x2x1xi8>, %arg1: tensor<2x3x3x1xi8>, %arg2: tensor<2xi32>, %arg3: tensor<1xi8>, %arg4: tensor<1xi8>) -> tensor<1x4x4x2xi32> {
  // CHECK: %[[TRANSPOSE_CONV2D:.*]] = spirv.Tosa.TransposeConv2D out_pad = [0, 0, 0, 0], stride = [1, 1], acc_type = <INT32>, local_bound = false, %arg0, %arg1, %arg2, %arg3, %arg4 : !spirv.arm.tensor<1x2x2x1xi8>, !spirv.arm.tensor<2x3x3x1xi8>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<1x4x4x2xi32>
  %res = tosa.transpose_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {out_pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, acc_type = i32, local_bound = false} : (tensor<1x2x2x1xi8>, tensor<2x3x3x1xi8>, tensor<2xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x4x4x2xi32>
  return %res : tensor<1x4x4x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clamp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @clamp_int
func.func @clamp_int(%arg0: tensor<4x8xi8>) -> tensor<4x8xi8> {
  // CHECK: %[[CLAMP:.*]] = spirv.Tosa.Clamp min_val = -2 : i8, max_val = 3 : i8, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<4x8xi8> -> !spirv.arm.tensor<4x8xi8>
  %res = tosa.clamp %arg0 {min_val = -2 : i8, max_val = 3 : i8, nan_mode = PROPAGATE} : (tensor<4x8xi8>) -> tensor<4x8xi8>
  return %res : tensor<4x8xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Erf
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @erf_fp
func.func @erf_fp(%arg0: tensor<47x38x51xf32>) -> tensor<47x38x51xf32> {
  // CHECK: %[[ERF:.*]] = spirv.Tosa.Erf  %arg0 : !spirv.arm.tensor<47x38x51xf32> -> !spirv.arm.tensor<47x38x51xf32>
  %res = tosa.erf %arg0  : (tensor<47x38x51xf32>) -> tensor<47x38x51xf32>
  return %res : tensor<47x38x51xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sigmoid
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sigmoid_fp
func.func @sigmoid_fp(%arg0: tensor<28x43x45xf32>) -> tensor<28x43x45xf32> {
  // CHECK: %[[SIGMOID:.*]] = spirv.Tosa.Sigmoid  %arg0 : !spirv.arm.tensor<28x43x45xf32> -> !spirv.arm.tensor<28x43x45xf32>
  %res = tosa.sigmoid %arg0  : (tensor<28x43x45xf32>) -> tensor<28x43x45xf32>
  return %res : tensor<28x43x45xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @tanh_fp
func.func @tanh_fp(%arg0: tensor<46x50x36xf16>) -> tensor<46x50x36xf16> {
  // CHECK: %[[TANH:.*]] = spirv.Tosa.Tanh  %arg0 : !spirv.arm.tensor<46x50x36xf16> -> !spirv.arm.tensor<46x50x36xf16>
  %res = tosa.tanh %arg0  : (tensor<46x50x36xf16>) -> tensor<46x50x36xf16>
  return %res : tensor<46x50x36xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Add
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @add_int
func.func @add_int(%arg0: tensor<4x7x3x10xi32>, %arg1: tensor<4x7x3x1xi32>) -> tensor<4x7x3x10xi32> {
  // CHECK: %[[ADD:.*]] = spirv.Tosa.Add  %arg0, %arg1 : !spirv.arm.tensor<4x7x3x10xi32>, !spirv.arm.tensor<4x7x3x1xi32> -> !spirv.arm.tensor<4x7x3x10xi32>
  %res = tosa.add %arg0, %arg1  : (tensor<4x7x3x10xi32>, tensor<4x7x3x1xi32>) -> tensor<4x7x3x10xi32>
  return %res : tensor<4x7x3x10xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ArithmeticRightShift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @arithmetic_right_shift_int
func.func @arithmetic_right_shift_int(%arg0: tensor<1x4xi16>, %arg1: tensor<3x4xi16>) -> tensor<3x4xi16> {
  // CHECK: %[[SHIFT:.*]] = spirv.Tosa.ArithmeticRightShift round = true, %arg0, %arg1 : !spirv.arm.tensor<1x4xi16>, !spirv.arm.tensor<3x4xi16> -> !spirv.arm.tensor<3x4xi16>
  %res = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<1x4xi16>, tensor<3x4xi16>) -> tensor<3x4xi16>
  return %res : tensor<3x4xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwiseand_int
func.func @bitwiseand_int(%arg0: tensor<4x1x7x12xi16>, %arg1: tensor<4x13x7x12xi16>) -> tensor<4x13x7x12xi16> {
  // CHECK: %[[BITWISEAND:.*]] = spirv.Tosa.BitwiseAnd  %arg0, %arg1 : !spirv.arm.tensor<4x1x7x12xi16>, !spirv.arm.tensor<4x13x7x12xi16> -> !spirv.arm.tensor<4x13x7x12xi16>
  %res = tosa.bitwise_and %arg0, %arg1  : (tensor<4x1x7x12xi16>, tensor<4x13x7x12xi16>) -> tensor<4x13x7x12xi16>
  return %res : tensor<4x13x7x12xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwiseor_int
func.func @bitwiseor_int(%arg0: tensor<11x30x23xi32>, %arg1: tensor<1x30x23xi32>) -> tensor<11x30x23xi32> {
  // CHECK: %[[BITWISEOR:.*]] = spirv.Tosa.BitwiseOr  %arg0, %arg1 : !spirv.arm.tensor<11x30x23xi32>, !spirv.arm.tensor<1x30x23xi32> -> !spirv.arm.tensor<11x30x23xi32>
  %res = tosa.bitwise_or %arg0, %arg1  : (tensor<11x30x23xi32>, tensor<1x30x23xi32>) -> tensor<11x30x23xi32>
  return %res : tensor<11x30x23xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwisexor_int
func.func @bitwisexor_int(%arg0: tensor<4x8x13x9xi16>, %arg1: tensor<4x8x1x9xi16>) -> tensor<4x8x13x9xi16> {
  // CHECK: %[[BITWISEXOR:.*]] = spirv.Tosa.BitwiseXor  %arg0, %arg1 : !spirv.arm.tensor<4x8x13x9xi16>, !spirv.arm.tensor<4x8x1x9xi16> -> !spirv.arm.tensor<4x8x13x9xi16>
  %res = tosa.bitwise_xor %arg0, %arg1  : (tensor<4x8x13x9xi16>, tensor<4x8x1x9xi16>) -> tensor<4x8x13x9xi16>
  return %res : tensor<4x8x13x9xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.IntDiv
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @intdiv_any
func.func @intdiv_any(%arg0: tensor<1x65533x1xi32>, %arg1: tensor<2x65533x1xi32>) -> tensor<2x65533x1xi32> {
  // CHECK: %[[INTDIV:.*]] = spirv.Tosa.IntDiv  %arg0, %arg1 : !spirv.arm.tensor<1x65533x1xi32>, !spirv.arm.tensor<2x65533x1xi32> -> !spirv.arm.tensor<2x65533x1xi32>
  %res = tosa.intdiv %arg0, %arg1  : (tensor<1x65533x1xi32>, tensor<2x65533x1xi32>) -> tensor<2x65533x1xi32>
  return %res : tensor<2x65533x1xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalAnd
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicaland_any
func.func @logicaland_any(%arg0: tensor<2x1x7x11xi1>, %arg1: tensor<2x4x7x11xi1>) -> tensor<2x4x7x11xi1> {
  // CHECK: %[[LOGICALAND:.*]] = spirv.Tosa.LogicalAnd  %arg0, %arg1 : !spirv.arm.tensor<2x1x7x11xi1>, !spirv.arm.tensor<2x4x7x11xi1> -> !spirv.arm.tensor<2x4x7x11xi1>
  %res = tosa.logical_and %arg0, %arg1  : (tensor<2x1x7x11xi1>, tensor<2x4x7x11xi1>) -> tensor<2x4x7x11xi1>
  return %res : tensor<2x4x7x11xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalLeftShift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalleftshift_any
func.func @logicalleftshift_any(%arg0: tensor<7x1x11x4xi8>, %arg1: tensor<7x8x11x4xi8>) -> tensor<7x8x11x4xi8> {
  // CHECK: %[[LOGICALLEFTSHIFT:.*]] = spirv.Tosa.LogicalLeftShift  %arg0, %arg1 : !spirv.arm.tensor<7x1x11x4xi8>, !spirv.arm.tensor<7x8x11x4xi8> -> !spirv.arm.tensor<7x8x11x4xi8>
  %res = tosa.logical_left_shift %arg0, %arg1  : (tensor<7x1x11x4xi8>, tensor<7x8x11x4xi8>) -> tensor<7x8x11x4xi8>
  return %res : tensor<7x8x11x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalRightShift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalrightshift_any
func.func @logicalrightshift_any(%arg0: tensor<6x13x1x19xi8>, %arg1: tensor<6x13x6x19xi8>) -> tensor<6x13x6x19xi8> {
  // CHECK: %[[LOGICALRIGHTSHIFT:.*]] = spirv.Tosa.LogicalRightShift  %arg0, %arg1 : !spirv.arm.tensor<6x13x1x19xi8>, !spirv.arm.tensor<6x13x6x19xi8> -> !spirv.arm.tensor<6x13x6x19xi8>
  %res = tosa.logical_right_shift %arg0, %arg1  : (tensor<6x13x1x19xi8>, tensor<6x13x6x19xi8>) -> tensor<6x13x6x19xi8>
  return %res : tensor<6x13x6x19xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalOr
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalor_any
func.func @logicalor_any(%arg0: tensor<3x6x12x5xi1>, %arg1: tensor<3x6x1x5xi1>) -> tensor<3x6x12x5xi1> {
  // CHECK: %[[LOGICALOR:.*]] = spirv.Tosa.LogicalOr  %arg0, %arg1 : !spirv.arm.tensor<3x6x12x5xi1>, !spirv.arm.tensor<3x6x1x5xi1> -> !spirv.arm.tensor<3x6x12x5xi1>
  %res = tosa.logical_or %arg0, %arg1  : (tensor<3x6x12x5xi1>, tensor<3x6x1x5xi1>) -> tensor<3x6x12x5xi1>
  return %res : tensor<3x6x12x5xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalXor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalxor_any
func.func @logicalxor_any(%arg0: tensor<11x4x9x12xi1>, %arg1: tensor<11x4x9x1xi1>) -> tensor<11x4x9x12xi1> {
  // CHECK: %[[LOGICALXOR:.*]] = spirv.Tosa.LogicalXor  %arg0, %arg1 : !spirv.arm.tensor<11x4x9x12xi1>, !spirv.arm.tensor<11x4x9x1xi1> -> !spirv.arm.tensor<11x4x9x12xi1>
  %res = tosa.logical_xor %arg0, %arg1  : (tensor<11x4x9x12xi1>, tensor<11x4x9x1xi1>) -> tensor<11x4x9x12xi1>
  return %res : tensor<11x4x9x12xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Maximum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @maximum_int
func.func @maximum_int(%arg0: tensor<1x2x65533x1xi32>, %arg1: tensor<1x2x65533x2xi32>) -> tensor<1x2x65533x2xi32> {
  // CHECK: %[[MAXIMUM:.*]] = spirv.Tosa.Maximum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<1x2x65533x1xi32>, !spirv.arm.tensor<1x2x65533x2xi32> -> !spirv.arm.tensor<1x2x65533x2xi32>
  %res = tosa.maximum %arg0, %arg1  {nan_mode = PROPAGATE} : (tensor<1x2x65533x1xi32>, tensor<1x2x65533x2xi32>) -> tensor<1x2x65533x2xi32>
  return %res : tensor<1x2x65533x2xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Minimum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @minimum_int
func.func @minimum_int(%arg0: tensor<15x2x10x11xi32>, %arg1: tensor<15x1x10x11xi32>) -> tensor<15x2x10x11xi32> {
  // CHECK: %[[MINIMUM:.*]] = spirv.Tosa.Minimum nan_mode = <Propagate>, %arg0, %arg1 : !spirv.arm.tensor<15x2x10x11xi32>, !spirv.arm.tensor<15x1x10x11xi32> -> !spirv.arm.tensor<15x2x10x11xi32>
  %res = tosa.minimum %arg0, %arg1  {nan_mode = PROPAGATE} : (tensor<15x2x10x11xi32>, tensor<15x1x10x11xi32>) -> tensor<15x2x10x11xi32>
  return %res : tensor<15x2x10x11xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Mul
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @mul_int
func.func @mul_int(%arg0: tensor<2x4xi32>, %arg1: tensor<2x1xi32>, %arg2: tensor<1xi8>) -> tensor<2x4xi32> {
  // CHECK: %[[MUL:.*]] = spirv.Tosa.Mul  %arg0, %arg1, %arg2 : !spirv.arm.tensor<2x4xi32>, !spirv.arm.tensor<2x1xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x4xi32>
  %res = tosa.mul %arg0, %arg1, %arg2 : (tensor<2x4xi32>, tensor<2x1xi32>, tensor<1xi8>) -> tensor<2x4xi32>
  return %res : tensor<2x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pow
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @pow_fp
func.func @pow_fp(%arg0: tensor<1x52x53xf16>, %arg1: tensor<44x52x53xf16>) -> tensor<44x52x53xf16> {
  // CHECK: %[[POW:.*]] = spirv.Tosa.Pow  %arg0, %arg1 : !spirv.arm.tensor<1x52x53xf16>, !spirv.arm.tensor<44x52x53xf16> -> !spirv.arm.tensor<44x52x53xf16>
  %res = tosa.pow %arg0, %arg1  : (tensor<1x52x53xf16>, tensor<44x52x53xf16>) -> tensor<44x52x53xf16>
  return %res : tensor<44x52x53xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sub
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sub_int
func.func @sub_int(%arg0: tensor<6x10x6x6xi32>, %arg1: tensor<1x10x6x6xi32>) -> tensor<6x10x6x6xi32> {
  // CHECK: %[[SUB:.*]] = spirv.Tosa.Sub  %arg0, %arg1 : !spirv.arm.tensor<6x10x6x6xi32>, !spirv.arm.tensor<1x10x6x6xi32> -> !spirv.arm.tensor<6x10x6x6xi32>
  %res = tosa.sub %arg0, %arg1  : (tensor<6x10x6x6xi32>, tensor<1x10x6x6xi32>) -> tensor<6x10x6x6xi32>
  return %res : tensor<6x10x6x6xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Table
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @table_int
func.func @table_int(%arg0: tensor<3x2xi8>, %arg1: tensor<256xi8>) -> tensor<3x2xi8> {
  // CHECK: %[[TABLE:.*]] = spirv.Tosa.Table  %arg0, %arg1 : !spirv.arm.tensor<3x2xi8>, !spirv.arm.tensor<256xi8> -> !spirv.arm.tensor<3x2xi8>
  %res = tosa.table %arg0, %arg1 : (tensor<3x2xi8>, tensor<256xi8>) -> tensor<3x2xi8>
  return %res : tensor<3x2xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Abs
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @abs_int
func.func @abs_int(%arg0: tensor<5x1x4x4xi32>) -> tensor<5x1x4x4xi32> {
  // CHECK: %[[ABS:.*]] = spirv.Tosa.Abs  %arg0 : !spirv.arm.tensor<5x1x4x4xi32> -> !spirv.arm.tensor<5x1x4x4xi32>
  %res = tosa.abs %arg0  : (tensor<5x1x4x4xi32>) -> tensor<5x1x4x4xi32>
  return %res : tensor<5x1x4x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.BitwiseNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @bitwisenot_int
func.func @bitwisenot_int(%arg0: tensor<12x56x50xi32>) -> tensor<12x56x50xi32> {
  // CHECK: %[[BITWISENOT:.*]] = spirv.Tosa.BitwiseNot  %arg0 : !spirv.arm.tensor<12x56x50xi32> -> !spirv.arm.tensor<12x56x50xi32>
  %res = tosa.bitwise_not %arg0  : (tensor<12x56x50xi32>) -> tensor<12x56x50xi32>
  return %res : tensor<12x56x50xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @ceil_fp
func.func @ceil_fp(%arg0: tensor<46x55x53xf16>) -> tensor<46x55x53xf16> {
  // CHECK: %[[CEIL:.*]] = spirv.Tosa.Ceil  %arg0 : !spirv.arm.tensor<46x55x53xf16> -> !spirv.arm.tensor<46x55x53xf16>
  %res = tosa.ceil %arg0  : (tensor<46x55x53xf16>) -> tensor<46x55x53xf16>
  return %res : tensor<46x55x53xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Clz
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @clz_int
func.func @clz_int(%arg0: tensor<14x10x7x5xi32>) -> tensor<14x10x7x5xi32> {
  // CHECK: %[[CLZ:.*]] = spirv.Tosa.Clz  %arg0 : !spirv.arm.tensor<14x10x7x5xi32> -> !spirv.arm.tensor<14x10x7x5xi32>
  %res = tosa.clz %arg0  : (tensor<14x10x7x5xi32>) -> tensor<14x10x7x5xi32>
  return %res : tensor<14x10x7x5xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cos
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @cos_fp
func.func @cos_fp(%arg0: tensor<44x49x51xf32>) -> tensor<44x49x51xf32> {
  // CHECK: %[[COS:.*]] = spirv.Tosa.Cos  %arg0 : !spirv.arm.tensor<44x49x51xf32> -> !spirv.arm.tensor<44x49x51xf32>
  %res = tosa.cos %arg0  : (tensor<44x49x51xf32>) -> tensor<44x49x51xf32>
  return %res : tensor<44x49x51xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Exp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @exp_fp
func.func @exp_fp(%arg0: tensor<37x53x47xf32>) -> tensor<37x53x47xf32> {
  // CHECK: %[[EXP:.*]] = spirv.Tosa.Exp  %arg0 : !spirv.arm.tensor<37x53x47xf32> -> !spirv.arm.tensor<37x53x47xf32>
  %res = tosa.exp %arg0  : (tensor<37x53x47xf32>) -> tensor<37x53x47xf32>
  return %res : tensor<37x53x47xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Floor
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @floor_fp
func.func @floor_fp(%arg0: tensor<40x52x42xf32>) -> tensor<40x52x42xf32> {
  // CHECK: %[[FLOOR:.*]] = spirv.Tosa.Floor  %arg0 : !spirv.arm.tensor<40x52x42xf32> -> !spirv.arm.tensor<40x52x42xf32>
  %res = tosa.floor %arg0  : (tensor<40x52x42xf32>) -> tensor<40x52x42xf32>
  return %res : tensor<40x52x42xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Log
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @log_fp
func.func @log_fp(%arg0: tensor<45x43x36xf16>) -> tensor<45x43x36xf16> {
  // CHECK: %[[LOG:.*]] = spirv.Tosa.Log  %arg0 : !spirv.arm.tensor<45x43x36xf16> -> !spirv.arm.tensor<45x43x36xf16>
  %res = tosa.log %arg0  : (tensor<45x43x36xf16>) -> tensor<45x43x36xf16>
  return %res : tensor<45x43x36xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.LogicalNot
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @logicalnot_any
func.func @logicalnot_any(%arg0: tensor<54x26x10xi1>) -> tensor<54x26x10xi1> {
  // CHECK: %[[LOGICALNOT:.*]] = spirv.Tosa.LogicalNot  %arg0 : !spirv.arm.tensor<54x26x10xi1> -> !spirv.arm.tensor<54x26x10xi1>
  %res = tosa.logical_not %arg0  : (tensor<54x26x10xi1>) -> tensor<54x26x10xi1>
  return %res : tensor<54x26x10xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Negate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @negate_int
func.func @negate_int(%arg0: tensor<2x3xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<2x3xi8> {
  // CHECK: %[[NEGATE:.*]] = spirv.Tosa.Negate  %arg0, %arg1, %arg2 : !spirv.arm.tensor<2x3xi8>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<2x3xi8>
  %res = tosa.negate %arg0, %arg1, %arg2 : (tensor<2x3xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x3xi8>
  return %res : tensor<2x3xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reciprocal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reciprocal_fp
func.func @reciprocal_fp(%arg0: tensor<38x47x44xf32>) -> tensor<38x47x44xf32> {
  // CHECK: %[[RECIPROCAL:.*]] = spirv.Tosa.Reciprocal  %arg0 : !spirv.arm.tensor<38x47x44xf32> -> !spirv.arm.tensor<38x47x44xf32>
  %res = tosa.reciprocal %arg0  : (tensor<38x47x44xf32>) -> tensor<38x47x44xf32>
  return %res : tensor<38x47x44xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rsqrt
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @rsqrt_fp
func.func @rsqrt_fp(%arg0: tensor<40x57x56xf32>) -> tensor<40x57x56xf32> {
  // CHECK: %[[RSQRT:.*]] = spirv.Tosa.Rsqrt  %arg0 : !spirv.arm.tensor<40x57x56xf32> -> !spirv.arm.tensor<40x57x56xf32>
  %res = tosa.rsqrt %arg0  : (tensor<40x57x56xf32>) -> tensor<40x57x56xf32>
  return %res : tensor<40x57x56xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Sin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @sin_fp
func.func @sin_fp(%arg0: tensor<49x38x58xf16>) -> tensor<49x38x58xf16> {
  // CHECK: %[[SIN:.*]] = spirv.Tosa.Sin  %arg0 : !spirv.arm.tensor<49x38x58xf16> -> !spirv.arm.tensor<49x38x58xf16>
  %res = tosa.sin %arg0  : (tensor<49x38x58xf16>) -> tensor<49x38x58xf16>
  return %res : tensor<49x38x58xf16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Select
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @select_int
func.func @select_int(%arg0: tensor<2x1xi1>, %arg1: tensor<2x4xi8>, %arg2: tensor<2x4xi8>) -> tensor<2x4xi8> {
  // CHECK: %[[SELECT:.*]] = spirv.Tosa.Select  %arg0, %arg1, %arg2 : !spirv.arm.tensor<2x1xi1>, !spirv.arm.tensor<2x4xi8>, !spirv.arm.tensor<2x4xi8> -> !spirv.arm.tensor<2x4xi8>
  %res = tosa.select %arg0, %arg1, %arg2 : (tensor<2x1xi1>, tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>
  return %res : tensor<2x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Equal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @equal_int
func.func @equal_int(%arg0: tensor<51x28x59xi32>, %arg1: tensor<51x1x59xi32>) -> tensor<51x28x59xi1> {
  // CHECK: %[[EQUAL:.*]] = spirv.Tosa.Equal  %arg0, %arg1 : !spirv.arm.tensor<51x28x59xi32>, !spirv.arm.tensor<51x1x59xi32> -> !spirv.arm.tensor<51x28x59xi1>
  %res = tosa.equal %arg0, %arg1  : (tensor<51x28x59xi32>, tensor<51x1x59xi32>) -> tensor<51x28x59xi1>
  return %res : tensor<51x28x59xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Greater
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @greater_int
func.func @greater_int(%arg0: tensor<11x10x10x2xi32>, %arg1: tensor<11x10x10x1xi32>) -> tensor<11x10x10x2xi1> {
  // CHECK: %[[GREATER:.*]] = spirv.Tosa.Greater  %arg0, %arg1 : !spirv.arm.tensor<11x10x10x2xi32>, !spirv.arm.tensor<11x10x10x1xi32> -> !spirv.arm.tensor<11x10x10x2xi1>
  %res = tosa.greater %arg0, %arg1  : (tensor<11x10x10x2xi32>, tensor<11x10x10x1xi32>) -> tensor<11x10x10x2xi1>
  return %res : tensor<11x10x10x2xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.GreaterEqual
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @greaterequal_int
func.func @greaterequal_int(%arg0: tensor<10x17x7x1xi32>, %arg1: tensor<10x17x7x16xi32>) -> tensor<10x17x7x16xi1> {
  // CHECK: %[[GREATEREQUAL:.*]] = spirv.Tosa.GreaterEqual  %arg0, %arg1 : !spirv.arm.tensor<10x17x7x1xi32>, !spirv.arm.tensor<10x17x7x16xi32> -> !spirv.arm.tensor<10x17x7x16xi1>
  %res = tosa.greater_equal %arg0, %arg1  : (tensor<10x17x7x1xi32>, tensor<10x17x7x16xi32>) -> tensor<10x17x7x16xi1>
  return %res : tensor<10x17x7x16xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAll
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_all
func.func @reduce_all(%arg0: tensor<2x3x4xi1>) -> tensor<2x1x4xi1> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceAll axis = 1, %arg0 : !spirv.arm.tensor<2x3x4xi1> -> !spirv.arm.tensor<2x1x4xi1>
  %res = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<2x3x4xi1>) -> tensor<2x1x4xi1>
  return %res : tensor<2x1x4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceAny
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_any
func.func @reduce_any(%arg0: tensor<2x3x4xi1>) -> tensor<2x1x4xi1> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceAny axis = 1, %arg0 : !spirv.arm.tensor<2x3x4xi1> -> !spirv.arm.tensor<2x1x4xi1>
  %res = tosa.reduce_any %arg0 {axis = 1 : i32} : (tensor<2x3x4xi1>) -> tensor<2x1x4xi1>
  return %res : tensor<2x1x4xi1>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMax
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_max_int
func.func @reduce_max_int(%arg0: tensor<2x3x4xi8>) -> tensor<2x1x4xi8> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceMax axis = 1, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x1x4xi8>
  %res = tosa.reduce_max %arg0 {axis = 1 : i32, nan_mode = PROPAGATE} : (tensor<2x3x4xi8>) -> tensor<2x1x4xi8>
  return %res : tensor<2x1x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceMin
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_min_int
func.func @reduce_min_int(%arg0: tensor<2x3x4xi8>) -> tensor<2x1x4xi8> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceMin axis = 1, nan_mode = <Propagate>, %arg0 : !spirv.arm.tensor<2x3x4xi8> -> !spirv.arm.tensor<2x1x4xi8>
  %res = tosa.reduce_min %arg0 {axis = 1 : i32, nan_mode = PROPAGATE} : (tensor<2x3x4xi8>) -> tensor<2x1x4xi8>
  return %res : tensor<2x1x4xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceProduct
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_product_fp
func.func @reduce_product_fp(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceProduct axis = 1, %arg0 : !spirv.arm.tensor<2x3x4xf32> -> !spirv.arm.tensor<2x1x4xf32>
  %res = tosa.reduce_product %arg0 {axis = 1 : i32} : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
  return %res : tensor<2x1x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.ReduceSum
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reduce_sum_int
func.func @reduce_sum_int(%arg0: tensor<2x3x4xi32>) -> tensor<2x1x4xi32> {
  // CHECK: %[[REDUCE:.*]] = spirv.Tosa.ReduceSum axis = 1, %arg0 : !spirv.arm.tensor<2x3x4xi32> -> !spirv.arm.tensor<2x1x4xi32>
  %res = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<2x3x4xi32>) -> tensor<2x1x4xi32>
  return %res : tensor<2x1x4xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Concat
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @concat_int
func.func @concat_int(%arg0: tensor<2x3xi8>, %arg1: tensor<2x4xi8>) -> tensor<2x7xi8> {
  // CHECK: %[[CONCAT:.*]] = spirv.Tosa.Concat axis = 1, %arg0, %arg1 : !spirv.arm.tensor<2x3xi8>, !spirv.arm.tensor<2x4xi8> -> !spirv.arm.tensor<2x7xi8>
  %res = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<2x3xi8>, tensor<2x4xi8>) -> tensor<2x7xi8>
  return %res : tensor<2x7xi8>
}

// -----

// CHECK-LABEL: spirv.ARM.Graph @concat_split_int
func.func @concat_split_int(%arg0: tensor<1xi8>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>, %arg3: tensor<1xi8>,
    %arg4: tensor<1xi8>, %arg5: tensor<1xi8>, %arg6: tensor<1xi8>, %arg7: tensor<1xi8>,
    %arg8: tensor<1xi8>, %arg9: tensor<1xi8>, %arg10: tensor<1xi8>, %arg11: tensor<1xi8>,
    %arg12: tensor<1xi8>, %arg13: tensor<1xi8>, %arg14: tensor<1xi8>, %arg15: tensor<1xi8>,
    %arg16: tensor<1xi8>, %arg17: tensor<1xi8>, %arg18: tensor<1xi8>, %arg19: tensor<1xi8>,
    %arg20: tensor<1xi8>, %arg21: tensor<1xi8>, %arg22: tensor<1xi8>, %arg23: tensor<1xi8>,
    %arg24: tensor<1xi8>, %arg25: tensor<1xi8>, %arg26: tensor<1xi8>, %arg27: tensor<1xi8>,
    %arg28: tensor<1xi8>, %arg29: tensor<1xi8>, %arg30: tensor<1xi8>, %arg31: tensor<1xi8>,
    %arg32: tensor<1xi8>, %arg33: tensor<1xi8>, %arg34: tensor<1xi8>, %arg35: tensor<1xi8>,
    %arg36: tensor<1xi8>, %arg37: tensor<1xi8>, %arg38: tensor<1xi8>, %arg39: tensor<1xi8>,
    %arg40: tensor<1xi8>, %arg41: tensor<1xi8>, %arg42: tensor<1xi8>, %arg43: tensor<1xi8>,
    %arg44: tensor<1xi8>, %arg45: tensor<1xi8>, %arg46: tensor<1xi8>, %arg47: tensor<1xi8>,
    %arg48: tensor<1xi8>, %arg49: tensor<1xi8>, %arg50: tensor<1xi8>, %arg51: tensor<1xi8>,
    %arg52: tensor<1xi8>, %arg53: tensor<1xi8>, %arg54: tensor<1xi8>, %arg55: tensor<1xi8>,
    %arg56: tensor<1xi8>, %arg57: tensor<1xi8>, %arg58: tensor<1xi8>, %arg59: tensor<1xi8>,
    %arg60: tensor<1xi8>, %arg61: tensor<1xi8>, %arg62: tensor<1xi8>, %arg63: tensor<1xi8>,
    %arg64: tensor<1xi8>) -> tensor<65xi8> {
  // CHECK: %[[CONCAT0:.*]] = spirv.Tosa.Concat axis = 0, %arg0, %arg1{{.*}} -> !spirv.arm.tensor<64xi8>
  // CHECK: %[[CONCAT1:.*]] = spirv.Tosa.Concat axis = 0, %[[CONCAT0]], %arg64 : !spirv.arm.tensor<64xi8>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<65xi8>
  %res = tosa.concat
      %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7,
      %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15,
      %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23,
      %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31,
      %arg32, %arg33, %arg34, %arg35, %arg36, %arg37, %arg38, %arg39,
      %arg40, %arg41, %arg42, %arg43, %arg44, %arg45, %arg46, %arg47,
      %arg48, %arg49, %arg50, %arg51, %arg52, %arg53, %arg54, %arg55,
      %arg56, %arg57, %arg58, %arg59, %arg60, %arg61, %arg62, %arg63,
      %arg64 {axis = 0 : i32}
      : (tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>,
      tensor<1xi8>) -> tensor<65xi8>
  return %res : tensor<65xi8>
}

// -----

// CHECK-LABEL: spirv.ARM.Graph @concat_split_dynamic
func.func @concat_split_dynamic(%arg0: tensor<?xi8>) -> tensor<?xi8> {
  // CHECK: %[[CONCAT0:.*]] = spirv.Tosa.Concat axis = 0, %arg0, %arg0{{.*}} -> !spirv.arm.tensor<?xi8>
  // CHECK: %[[CONCAT1:.*]] = spirv.Tosa.Concat axis = 0, %[[CONCAT0]], %arg0 : !spirv.arm.tensor<?xi8>, !spirv.arm.tensor<?xi8> -> !spirv.arm.tensor<?xi8>
  %res = tosa.concat %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0, %arg0 {axis = 0 : i32} : (tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>, tensor<?xi8>) -> tensor<?xi8>
  return %res : tensor<?xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Pad
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @pad_int
func.func @pad_int(%arg0: tensor<1x2xi8>) -> tensor<4x8xi8> {
  %padding = "tosa.const_shape"() <{values = dense<[1, 2, 3, 3]> : tensor<4xindex>}> : () -> !tosa.shape<4>
  %pad_const = "tosa.const"() <{values = dense<7> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK: %[[PADDING:.*]] = spirv.Constant dense<[1, 2, 3, 3]> : !spirv.arm.tensor<4xi32>
  // CHECK: %[[PAD_CONST:.*]] = spirv.Constant dense<7> : !spirv.arm.tensor<1xi8>
  // CHECK: %[[PAD:.*]] = spirv.Tosa.Pad  %arg0, %[[PADDING]], %[[PAD_CONST]] : !spirv.arm.tensor<1x2xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<1xi8> -> !spirv.arm.tensor<4x8xi8>
  %res = tosa.pad %arg0, %padding, %pad_const : (tensor<1x2xi8>, !tosa.shape<4>, tensor<1xi8>) -> tensor<4x8xi8>
  return %res : tensor<4x8xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reshape_int
func.func @reshape_int(%arg0: tensor<25x6x29x35xi16>) -> tensor<125x6x7x29xi16> {
  %shape = "tosa.const_shape"() <{values = dense<[125, 6, 7, 29]> : tensor<4xindex>}> : () -> !tosa.shape<4>
  // CHECK: %[[SHAPE:.*]] = spirv.Constant dense<[125, 6, 7, 29]> : !spirv.arm.tensor<4xi32>
  // CHECK: %[[RESHAPE:.*]] = spirv.Tosa.Reshape  %arg0, %[[SHAPE]] : !spirv.arm.tensor<25x6x29x35xi16>, !spirv.arm.tensor<4xi32> -> !spirv.arm.tensor<125x6x7x29xi16>
  %res = tosa.reshape %arg0, %shape : (tensor<25x6x29x35xi16>, !tosa.shape<4>) -> tensor<125x6x7x29xi16>
  return %res : tensor<125x6x7x29xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Reverse
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @reverse_int
func.func @reverse_int(%arg0: tensor<20x5x28x31xi32>) -> tensor<20x5x28x31xi32> {
  // CHECK: %[[REVERSE:.*]] = spirv.Tosa.Reverse axis = 2, %arg0 : !spirv.arm.tensor<20x5x28x31xi32> -> !spirv.arm.tensor<20x5x28x31xi32>
  %res = tosa.reverse %arg0 {axis = 2 : i32} : (tensor<20x5x28x31xi32>) -> tensor<20x5x28x31xi32>
  return %res : tensor<20x5x28x31xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Slice
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @slice_int
func.func @slice_int(%arg0: tensor<32x19x41xi8>) -> tensor<21x5x2xi8> {
  %start = "tosa.const_shape"() <{values = dense<[8, 11, 39]> : tensor<3xindex>}> : () -> !tosa.shape<3>
  %size = "tosa.const_shape"() <{values = dense<[21, 5, 2]> : tensor<3xindex>}> : () -> !tosa.shape<3>
  // CHECK: %[[START:.*]] = spirv.Constant dense<[8, 11, 39]> : !spirv.arm.tensor<3xi32>
  // CHECK: %[[SIZE:.*]] = spirv.Constant dense<[21, 5, 2]> : !spirv.arm.tensor<3xi32>
  // CHECK: %[[SLICE:.*]] = spirv.Tosa.Slice  %arg0, %[[START]], %[[SIZE]] : !spirv.arm.tensor<32x19x41xi8>, !spirv.arm.tensor<3xi32>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<21x5x2xi8>
  %res = tosa.slice %arg0, %start, %size : (tensor<32x19x41xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<21x5x2xi8>
  return %res : tensor<21x5x2xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Tile
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @tile_int
func.func @tile_int(%arg0: tensor<10x28x21xi16>) -> tensor<10x28x63xi16> {
  %multiples = "tosa.const_shape"() <{values = dense<[1, 1, 3]> : tensor<3xindex>}> : () -> !tosa.shape<3>
  // CHECK: %[[MULTIPLES:.*]] = spirv.Constant dense<[1, 1, 3]> : !spirv.arm.tensor<3xi32>
  // CHECK: %[[TILE:.*]] = spirv.Tosa.Tile  %arg0, %[[MULTIPLES]] : !spirv.arm.tensor<10x28x21xi16>, !spirv.arm.tensor<3xi32> -> !spirv.arm.tensor<10x28x63xi16>
  %res = tosa.tile %arg0, %multiples : (tensor<10x28x21xi16>, !tosa.shape<3>) -> tensor<10x28x63xi16>
  return %res : tensor<10x28x63xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Transpose
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @transpose_int
func.func @transpose_int(%arg0: tensor<14x28x1x61xi16>) -> tensor<1x14x28x61xi16> {
  // CHECK: %[[TRANSPOSE:.*]] = spirv.Tosa.Transpose perms = [2, 0, 1, 3], %arg0 : !spirv.arm.tensor<14x28x1x61xi16> -> !spirv.arm.tensor<1x14x28x61xi16>
  %res = tosa.transpose %arg0 {perms = array<i32: 2, 0, 1, 3>} : (tensor<14x28x1x61xi16>) -> tensor<1x14x28x61xi16>
  return %res : tensor<1x14x28x61xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Gather
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @gather_int
func.func @gather_int(%arg0: tensor<31x11x45xi32>, %arg1: tensor<31x15xi32>) -> tensor<31x15x45xi32> {
  // CHECK: %[[GATHER:.*]] = spirv.Tosa.Gather  %arg0, %arg1 : !spirv.arm.tensor<31x11x45xi32>, !spirv.arm.tensor<31x15xi32> -> !spirv.arm.tensor<31x15x45xi32>
  %res = tosa.gather %arg0, %arg1 : (tensor<31x11x45xi32>, tensor<31x15xi32>) -> tensor<31x15x45xi32>
  return %res : tensor<31x15x45xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Scatter
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @scatter_int
func.func @scatter_int(%arg0: tensor<34x28x54xi32>, %arg1: tensor<34x18xi32>, %arg2: tensor<34x18x54xi32>) -> tensor<34x28x54xi32> {
  // CHECK: %[[SCATTER:.*]] = spirv.Tosa.Scatter  %arg0, %arg1, %arg2 : !spirv.arm.tensor<34x28x54xi32>, !spirv.arm.tensor<34x18xi32>, !spirv.arm.tensor<34x18x54xi32> -> !spirv.arm.tensor<34x28x54xi32>
  %res = tosa.scatter %arg0, %arg1, %arg2 : (tensor<34x28x54xi32>, tensor<34x18xi32>, tensor<34x18x54xi32>) -> tensor<34x28x54xi32>
  return %res : tensor<34x28x54xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Resize
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @resize_int
func.func @resize_int(%arg0: tensor<1x1x31x55xi8>) -> tensor<1x1x278x55xi8> {
  %scale = "tosa.const_shape"() <{values = dense<[16, 1, 9, 1]> : tensor<4xindex>}> : () -> !tosa.shape<4>
  %offset = "tosa.const_shape"() <{values = dense<0> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %border = "tosa.const_shape"() <{values = dense<[0, 7]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  // CHECK: %[[SCALE:.*]] = spirv.Constant dense<[16, 1, 9, 1]> : !spirv.arm.tensor<4xi32>
  // CHECK: %[[OFFSET:.*]] = spirv.Constant dense<0> : !spirv.arm.tensor<2xi32>
  // CHECK: %[[BORDER:.*]] = spirv.Constant dense<[0, 7]> : !spirv.arm.tensor<2xi32>
  // CHECK: %[[RESIZE:.*]] = spirv.Tosa.Resize mode = <NearestNeighbor>, %arg0, %[[SCALE]], %[[OFFSET]], %[[BORDER]] : !spirv.arm.tensor<1x1x31x55xi8>, !spirv.arm.tensor<4xi32>, !spirv.arm.tensor<2xi32>, !spirv.arm.tensor<2xi32> -> !spirv.arm.tensor<1x1x278x55xi8>
  %res = tosa.resize %arg0, %scale, %offset, %border {mode = NEAREST_NEIGHBOR} : (tensor<1x1x31x55xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x1x278x55xi8>
  return %res : tensor<1x1x278x55xi8>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @cast_int
func.func @cast_int(%arg0: tensor<2x3xi8>) -> tensor<2x3xi32> {
  // CHECK: %[[CAST:.*]] = spirv.Tosa.Cast  %arg0 : !spirv.arm.tensor<2x3xi8> -> !spirv.arm.tensor<2x3xi32>
  %res = tosa.cast %arg0 : (tensor<2x3xi8>) -> tensor<2x3xi32>
  return %res : tensor<2x3xi32>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Rescale
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @rescale_int
func.func @rescale_int(%arg0: tensor<2x3xi16>) -> tensor<2x3xi16> {
  %multiplier = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  %shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
  %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  // CHECK: %[[MULTIPLIER:.*]] = spirv.Constant dense<1073741824> : !spirv.arm.tensor<1xi32>
  // CHECK: %[[SHIFT:.*]] = spirv.Constant dense<30> : !spirv.arm.tensor<1xi8>
  // CHECK: %[[INPUT_ZP:.*]] = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // CHECK: %[[OUTPUT_ZP:.*]] = spirv.Constant dense<0> : !spirv.arm.tensor<1xi16>
  // CHECK: %[[RESCALE:.*]] = spirv.Tosa.Rescale scale32 = true, rounding_mode = <DoubleRound>, per_channel = false, input_unsigned = false, output_unsigned = false, %arg0, %[[MULTIPLIER]], %[[SHIFT]], %[[INPUT_ZP]], %[[OUTPUT_ZP]] : !spirv.arm.tensor<2x3xi16>, !spirv.arm.tensor<1xi32>, !spirv.arm.tensor<1xi8>, !spirv.arm.tensor<1xi16>, !spirv.arm.tensor<1xi16> -> !spirv.arm.tensor<2x3xi16>
  %res = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {scale32 = true, rounding_mode = DOUBLE_ROUND, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<2x3xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi16>) -> tensor<2x3xi16>
  return %res : tensor<2x3xi16>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Const
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @const_int
func.func @const_int() -> tensor<2x3xi8> {
  // CHECK: %[[CONST:.*]] = spirv.Constant dense<{{.*}}> : !spirv.arm.tensor<2x3xi8>
  %res = "tosa.const"() <{values = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi8>}> : () -> tensor<2x3xi8>
  return %res : tensor<2x3xi8>
}

// -----

// CHECK-LABEL: spirv.ARM.Graph @const_i4
func.func @const_i4() -> tensor<2xi4> {
  // CHECK: %[[CONST:.*]] = spirv.Constant dense<[7, -8]> : !spirv.arm.tensor<2xi8>
  %res = "tosa.const"() <{values = dense<[7, -8]> : tensor<2xi4>}> : () -> tensor<2xi4>
  return %res : tensor<2xi4>
}

// -----

// CHECK-LABEL: spirv.ARM.Graph @const_i48
func.func @const_i48() -> tensor<2xi48> {
  // CHECK: %[[CONST:.*]] = spirv.Constant dense<[140737488355327, -140737488355328]> : !spirv.arm.tensor<2xi64>
  %res = "tosa.const"() <{values = dense<[140737488355327, -140737488355328]> : tensor<2xi48>}> : () -> tensor<2xi48>
  return %res : tensor<2xi48>
}

// -----

// CHECK-LABEL: spirv.ARM.Graph @const_shape_empty
func.func @const_shape_empty() -> !tosa.shape<0> {
  // CHECK: %[[SHAPE:.*]] = spirv.Constant dense<1> : !spirv.arm.tensor<1xi32>
  %res = "tosa.const_shape"() <{values = dense<> : tensor<0xindex>}> : () -> !tosa.shape<0>
  return %res : !tosa.shape<0>
}

// -----

//===----------------------------------------------------------------------===//
// spirv.TOSA.Identity
//===----------------------------------------------------------------------===//

// CHECK-LABEL: spirv.ARM.Graph @identity_int
func.func @identity_int(%arg0: tensor<2x3xi8>) -> tensor<2x3xi8> {
  // CHECK-NOT: spirv.Tosa.Identity
  // CHECK: spirv.ARM.GraphOutputs %arg0 : !spirv.arm.tensor<2x3xi8>
  %res = tosa.identity %arg0 : (tensor<2x3xi8>) -> tensor<2x3xi8>
  return %res : tensor<2x3xi8>
}

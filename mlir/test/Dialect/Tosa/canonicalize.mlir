// RUN: mlir-opt --split-input-file -canonicalize="test-convergence" %s | FileCheck %s

// CHECK-LABEL: @argmax_nofold
func.func @argmax_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xi32> {
  // CHECK: tosa.argmax
  %0 = tosa.argmax %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xi32>
  return %0 : tensor<?x1xi32>
}

// -----

// CHECK-LABEL: @add_bcast_zero_int
func.func @add_bcast_zero_int(%arg0: tensor<4x2x3xi32>) -> tensor<4x2x3xi32> {
  // CHECK-NOT: tosa.add
  // CHECK: return %arg0
  %zeros = "tosa.const"() {value = dense<0> : tensor<1x1x1xi32>} : () -> tensor<1x1x1xi32>
  %1 = tosa.add %arg0, %zeros : (tensor<4x2x3xi32>, tensor<1x1x1xi32>) -> tensor<4x2x3xi32>
  return %1 : tensor<4x2x3xi32>
}

// -----

// CHECK-LABEL: @add_zero_int
func.func @add_zero_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.add
  %zeros = "tosa.const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = tosa.add %arg0, %zeros : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @cast_fold
func.func @cast_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.cast %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @cast_nofold
func.func @cast_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xi32> {
  // CHECK: tosa.cast
  %0 = tosa.cast %arg0 : (tensor<?x1xf32>) -> tensor<?x1xi32>
  return %0 : tensor<?x1xi32>
}

// -----

// CHECK-LABEL: @clamp_i32_not_noop
func.func @clamp_i32_not_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = 1 : i64, max_int = 4 : i64, min_fp = 1.0 : f32, max_fp = 4.0 : f32} : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @clamp_f16_not_noop
func.func @clamp_f16_not_noop(%arg0: tensor<4xf16>) -> tensor<4xf16> {
  // CHECK: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = -128 : i64, max_int = 127 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} : (tensor<4xf16>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// CHECK-LABEL: @clamp_f32_not_noop
func.func @clamp_f32_not_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = -128 : i64, max_int = 127 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_f16_is_noop
func.func @clamp_f16_is_noop(%arg0: tensor<4xf16>) -> tensor<4xf16> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  // 0xFF800000 and 0x7F800000 are respectively negative and positive F32 infinity.
  %0 = tosa.clamp %arg0 {min_int = -128 : i64, max_int = 127 : i64, min_fp = 0xFF800000 : f32, max_fp = 0x7F800000 : f32} : (tensor<4xf16>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// CHECK-LABEL: @clamp_f32_is_noop
func.func @clamp_f32_is_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  // 0xFF800000 and 0x7F800000 are respectively negative and positive F32 infinity.
  %0 = tosa.clamp %arg0 {min_int = -128 : i64, max_int = 127 : i64, min_fp = 0xFF800000 : f32, max_fp = 0x7F800000 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_int8_is_noop
func.func @clamp_int8_is_noop(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = -128 : i64, max_int = 127 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @clamp_int16_is_noop
func.func @clamp_int16_is_noop(%arg0: tensor<4xi16>) -> tensor<4xi16> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = -32768 : i64, max_int = 32767 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xi16>) -> tensor<4xi16>
  return %0 : tensor<4xi16>
}

// -----

// CHECK-LABEL: @clamp_uint8_is_noop
func.func @clamp_uint8_is_noop(%arg0: tensor<4xui8>) -> tensor<4xui8> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_int = 0 : i64, max_int = 255 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}

// -----

// CHECK-LABEL: @clamp_twice_is_single_clamp
func.func @clamp_twice_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_fp = 3.000000e+00 : f32, max_int = 2 : i64, min_fp = -3.000000e+00 : f32, min_int = -2 : i64}
  %0 = tosa.clamp %arg0 {max_fp = 3.0 : f32, max_int = 4 : i64, min_fp = -5.0 : f32, min_int = -2 : i64} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_fp = 5.0 : f32, max_int = 2 : i64, min_fp = -3.0 : f32, min_int = -4 : i64} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @concat_fold
func.func @concat_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.concat %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @concat_fold_cast
func.func @concat_fold_cast(%arg0: tensor<?x1xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[VAR0:.*]] = tensor.cast %arg0
  // CHECK: return %[[VAR0]]
  %0 = tosa.concat %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @conv2d_stride_2
func.func @conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>) -> tensor<4x10x10x3xf32> {
  // CHECK: tosa.conv2d
  %weight = "tosa.const"() {value = dense<[[[[1.0, 1.0]]], [[[1.0, 1.0]]], [[[1.0, 1.0]]]]> : tensor<3x1x1x2xf32>} : ()-> tensor<3x1x1x2xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<3xf32>} : ()-> tensor<3xf32>
  %0 = tosa.conv2d %arg0, %weight, %bias {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<4x10x10x3xf32>
  return %0 : tensor<4x10x10x3xf32>
}

// -----

// CHECK-LABEL: @conv2d_weight_2x2
func.func @conv2d_weight_2x2(%arg0: tensor<4x10x10x1xf32>) -> tensor<4x10x10x1xf32> {
  // CHECK: tosa.conv2d
  %weight = "tosa.const"() {value = dense<[[[[1.0], [1.0]], [[1.0], [1.0]]]]> : tensor<1x2x2x1xf32>} : ()-> tensor<1x2x2x1xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : ()-> tensor<1xf32>
  %0 = tosa.conv2d %arg0, %weight, %bias {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>) -> tensor<4x10x10x1xf32>
  return %0 : tensor<4x10x10x1xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_stride_2
func.func @depthwise_conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: tosa.depthwise_conv2d
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_weight_2x2
func.func @depthwise_conv2d_weight_2x2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: tosa.depthwise_conv2d
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @max_pool2d_is_noop
func.func @max_pool2d_is_noop(%arg0: tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32> {
  // CHECK-NOT: tosa.max_pool2d
  // CHECK: return %arg0
  %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32>
  return %0 : tensor<10x1x1x3xf32>
}

// -----

// CHECK-LABEL: @pad_noop
func.func @pad_noop(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const"() { value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = tosa.pad %arg0, %0 : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_i32
func.func @pad_determine_val_i32(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{value = dense<0> : tensor<i32>}
  // CHECK: tosa.pad %arg0, %arg1, %[[ZERO]]
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = tosa.pad %arg0, %arg1 : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @pad_determine_val_f32
func.func @pad_determine_val_f32(%arg0: tensor<?x?xf32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xf32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: tosa.pad %arg0, %arg1, %[[ZERO]]
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = tosa.pad %arg0, %arg1 : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_quant
func.func @pad_determine_val_quant(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() <{value = dense<42> : tensor<i32>}
  // CHECK: tosa.pad %arg0, %arg1, %[[ZERO]]
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = tosa.pad %arg0, %arg1 {quantization_info = #tosa.pad_quant<input_zp = 42>} : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @mul_one_float
func.func @mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = tosa.mul %arg0, %ones {shift = 0 : i8} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_bcast_one_float
func.func @mul_bcast_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %1 = tosa.mul %ones, %arg0 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_one_int
func.func @mul_one_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = tosa.mul %arg0, %ones {shift = 0 : i8} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @mul_zero_broadcast
func.func @mul_zero_broadcast(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: %[[ZERO:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x3xf32>}
  // CHECK-NOT: tosa.mul
  %zeros = "tosa.const"() {value = dense<0.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %1 = tosa.mul %arg0, %zeros {shift = 0 : i8} : (tensor<2x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>

  // CHECK-NOT: tosa.mul
  // CHECK: return %[[ZERO]], %[[ZERO]]
  %2 = tosa.mul %zeros, %arg0 {shift = 0 : i8} : (tensor<1x1xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @select_same_value
func.func @select_same_value(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = tosa.select %arg0, %arg1, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @select_true_value
func.func @select_true_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c1 = "tosa.const"() {value = dense<1> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c1, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @select_false_value
func.func @select_false_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = "tosa.const"() {value = dense<0> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c0, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @select_not_pred
func.func @select_not_pred(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = tosa.logical_not %arg0 : (tensor<2x3xi1>) -> tensor<2x3xi1>
  %1 = tosa.select %0, %arg1, %arg2 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: tosa.select %arg0, %arg2, %arg1
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @reduce_all_fold
func.func @reduce_all_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_all %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_all_nofold
func.func @reduce_all_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_all
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_any_fold
func.func @reduce_any_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_any %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_any_nofold
func.func @reduce_any_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_any
  %0 = tosa.reduce_any %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_max_fold
func.func @reduce_max_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_max %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_max_nofold
func.func @reduce_max_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_max
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_min_fold
func.func @reduce_min_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_min %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_min_nofold
func.func @reduce_min_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_min
  %0 = tosa.reduce_min %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_prod_fold
func.func @reduce_prod_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_prod %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_prod_nofold
func.func @reduce_prod_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_prod
  %0 = tosa.reduce_prod %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_sum_fold
func.func @reduce_sum_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_sum %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_sum_nofold
func.func @reduce_sum_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_sum
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize
func.func @reshape_canonicalize(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  // CHECK: return %arg0
  %0 = tosa.reshape %arg0 {new_shape = array<i64: -1, 10>}: (tensor<?x10xf32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_dyn_nofold
func.func @reshape_canonicalize_dyn_nofold(%arg0: tensor<?x?x10xf32>) -> tensor<?x?x10xf32> {
  // CHECK: %[[VAR0:.+]] = tosa.reshape %arg0 {new_shape = array<i64: -1, 2, 10>} : (tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
  // CHECK: return %[[VAR0]] : tensor<?x?x10xf32>
  %0 = tosa.reshape %arg0 {new_shape = array<i64: -1, 2, 10>} : (tensor<?x?x10xf32>) -> tensor<?x?x10xf32>
  return %0 : tensor<?x?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_double
func.func @reshape_canonicalize_double(%arg0: tensor<?x10xf32>) -> tensor<?x5xf32> {
  // CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0 {new_shape = array<i64: -1, 5>}
  // CHECK: return %[[VAL_1]]
  %0 = tosa.reshape %arg0 {new_shape = array<i64: 5, -1>}: (tensor<?x10xf32>) -> tensor<5x?xf32>
  %1 = tosa.reshape %0 {new_shape = array<i64: -1, 5>}: (tensor<5x?xf32>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const
func.func @reshape_canonicalize_const() -> tensor<1x5xi32> {
  // CHECK: %[[VAR0:.+]] = "tosa.const"() <{value = dense<{{\[\[}}0, 1, 2, 3, 4]]> : tensor<1x5xi32>}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 5>} : (tensor<5xi32>) -> tensor<1x5xi32>
  return %1 : tensor<1x5xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_dynamic
func.func @reshape_canonicalize_const_dynamic() -> tensor<1x?xi32> {
  // CHECK: tosa.reshape
  %0 = "tosa.const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 5>} : (tensor<5xi32>) -> tensor<1x?xi32>
  return %1 : tensor<1x?xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_splat
func.func @reshape_canonicalize_const_splat() -> (tensor<10xi32>, tensor<1x10xi32>) {
  // CHECK-DAG: %[[VAR0:.+]] = "tosa.const"() <{value = dense<0> : tensor<10xi32>}
  // CHECK-DAG: %[[VAR1:.+]] = "tosa.const"() <{value = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]], %[[VAR1]]
  %0 = "tosa.const"() {value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 10>} : (tensor<10xi32>) -> tensor<1x10xi32>
  return %0 , %1 : tensor<10xi32>, tensor<1x10xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_sparse
func.func @reshape_canonicalize_const_sparse() -> (tensor<3xi32>, tensor<1x3xi32>) {
  // CHECK: tosa.reshape
  %0 = "tosa.const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : ()-> tensor<3xi32>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 3>} : (tensor<3xi32>) -> tensor<1x3xi32>
  return %0 , %1 : tensor<3xi32>, tensor<1x3xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_quant_nofold
func.func @reshape_canonicalize_quant_nofold() -> (tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>) {
  // disabled folding for quantized element types
  // CHECK{LITERAL}: "tosa.const"() <{value = dense<[1, 2, 3]> : tensor<3xi8>}> : () -> tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>
  // CHECK{LITERAL}: tosa.reshape %0 {new_shape = array<i64: 1, 3>} : (tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %0 = "tosa.const"() {value = dense<[1, 2, 3]> : tensor<3xi8>} : ()-> tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1, 3>} : (tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  return %1 :  tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: @transpose_canonicalize_strip_quant
func.func @transpose_canonicalize_strip_quant() -> (tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) {
  // CHECK: "tosa.const"() <{value = dense<0> : tensor<1x2x3xi8>}> : () -> tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>
  // CHECK: tosa.reshape %0 {new_shape = array<i64: 2, 1, 3>} : (tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %perms = "tosa.const"() {value = dense<[1, 0, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %0 = "tosa.const"() {value = dense<0> : tensor<1x2x3xi8>} : ()-> tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %1 = tosa.transpose %0, %perms : (tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>, tensor<3xi32>) -> tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  return %1 :  tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: @slice_fold
func.func @slice_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = tosa.slice %arg0 { size = array<i64: 3, 4>, start = array<i64: 0, 0>}: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @slice_nofold
func.func @slice_nofold(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: tosa.slice
  %0 = tosa.slice %arg0 { size = array<i64: 3, 4>, start = array<i64: 0, 0>}: (tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tile_fold
func.func @tile_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = tosa.tile %arg0 { multiples = array<i64: 1, 1> }: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @tile_nofold
func.func @tile_nofold(%arg0: tensor<3x4xf32>) -> tensor<3x8xf32> {
  // CHECK: tosa.tile
  %0 = tosa.tile %arg0 { multiples = array<i64: 1, 2> }: (tensor<3x4xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// -----

// CHECK-LABEL: @transpose_no_op
func.func @transpose_no_op(%arg0: tensor<3x4x5x6xf32>) -> tensor<3x4x5x6xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.transpose
  %perms = "tosa.const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = tosa.transpose %arg0, %perms : (tensor<3x4x5x6xf32>, tensor<4xi32>) -> tensor<3x4x5x6xf32>
  return %1 : tensor<3x4x5x6xf32>
}

// -----

// CHECK-LABEL: @transpose_is_reshape
func.func @transpose_is_reshape(%arg0: tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32> {
  // CHECK: tosa.reshape %arg0 {new_shape = array<i64: 1, 4, 1, 5>} : (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  %perms = "tosa.const"() <{value = dense<[3, 1, 0, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %0 = tosa.transpose %arg0, %perms : (tensor<1x4x5x1xf32>, tensor<4xi32>) -> tensor<1x4x1x5xf32>
  return %0 : tensor<1x4x1x5xf32>
}

// -----

// CHECK-LABEL: @single_bit_reshape
// https://github.com/llvm/llvm-project/issues/55440
func.func @single_bit_reshape() -> tensor<1xi1> {
  // CHECK: "tosa.const"() <{value = dense<true> : tensor<1xi1>}
  %0 = arith.constant dense<true> : tensor<1x1xi1>
  %1 = tosa.reshape %0 {new_shape = array<i64: 1>} : (tensor<1x1xi1>) -> tensor<1xi1>
  return %1 : tensor<1xi1>
}

// -----

// CHECK-LABEL: @fold_resize_nearest
func.func @fold_resize_nearest(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %resize = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR" , scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

// -----

// CHECK-LABEL: @fold_resize_bilinear
func.func @fold_resize_bilinear(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %resize = tosa.resize %arg0 {mode = "BILINEAR" , scale = array<i64: 2, 2, 1, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_final_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12x1xf32>, %[[VAL_1:.*]]: tensor<1x12x12x1xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]] : tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>
func.func @canonicalize_concat_slice_final_axis(%arg0 : tensor<1x12x12x1xf32>, %arg1 : tensor<1x12x12x1xf32>) -> (tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 3 : i32} : (tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>) -> tensor<1x12x12x2xf32>
  %1 = tosa.slice %0 {size = array<i64: 1, 12, 12, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<1x12x12x2xf32>) -> tensor<1x12x12x1xf32>
  %2 = tosa.slice %0 {size = array<i64: 1, 12, 12, 1>, start = array<i64: 0, 0, 0, 1>} : (tensor<1x12x12x2xf32>) -> tensor<1x12x12x1xf32>
  return %1, %2 : tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_middle_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]] : tensor<1x12x12xf32>, tensor<1x12x12xf32>
func.func @canonicalize_concat_slice_middle_axis(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x12x12xf32>, tensor<1x12x12xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x24x12xf32>
  %1 = tosa.slice %0 {size = array<i64: 1, 12, 12>, start = array<i64: 0, 0, 0>} : (tensor<1x24x12xf32>) -> tensor<1x12x12xf32>
  %2 = tosa.slice %0 {size = array<i64: 1, 12, 12>, start = array<i64: 0, 12, 0>} : (tensor<1x24x12xf32>) -> tensor<1x12x12xf32>
  return %1, %2 : tensor<1x12x12xf32>, tensor<1x12x12xf32>
}

// -----

// CHECK-LABEL: @canonicalize_cross_concat_inputs
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK: %[[VAL_2:.*]] = tosa.concat %[[VAL_0]], %[[VAL_1]] {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
// CHECK: %[[VAL_3:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 1, 12, 15>, start = array<i64: 0, 0, 0>} : (tensor<1x12x24xf32>) -> tensor<1x12x15xf32>
// CHECK: %[[VAL_4:.*]] = tosa.slice %[[VAL_2]] {size = array<i64: 1, 12, 20>, start = array<i64: 0, 0, 4>} : (tensor<1x12x24xf32>) -> tensor<1x12x20xf32>
// CHECK: return %[[VAL_3]], %[[VAL_4]] : tensor<1x12x15xf32>, tensor<1x12x20xf32>
func.func @canonicalize_cross_concat_inputs(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x12x15xf32>, tensor<1x12x20xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
  %1 = tosa.slice %0 {size = array<i64: 1, 12, 15>, start = array<i64: 0, 0, 0>} : (tensor<1x12x24xf32>) -> tensor<1x12x15xf32>
  %2 = tosa.slice %0 {size = array<i64: 1, 12, 20>, start = array<i64: 0, 0, 4>} : (tensor<1x12x24xf32>) -> tensor<1x12x20xf32>
  return %1, %2 : tensor<1x12x15xf32>, tensor<1x12x20xf32>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_on_non_concat_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK: %[[VAL_2:.*]] = tosa.slice %[[VAL_0]] {size = array<i64: 1, 6, 12>, start = array<i64: 0, 0, 0>} : (tensor<1x12x12xf32>) -> tensor<1x6x12xf32>
// CHECK: %[[VAL_3:.*]] = tosa.slice %[[VAL_1]] {size = array<i64: 1, 3, 12>, start = array<i64: 1, 3, 0>} : (tensor<1x12x12xf32>) -> tensor<1x3x12xf32>
// CHECK: return %[[VAL_2]], %[[VAL_3]] : tensor<1x6x12xf32>, tensor<1x3x12xf32>
func.func @canonicalize_concat_slice_on_non_concat_axis(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x6x12xf32>, tensor<1x3x12xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
  %1 = tosa.slice %0 {size = array<i64: 1, 6, 12>, start = array<i64: 0, 0, 0>} : (tensor<1x12x24xf32>) -> tensor<1x6x12xf32>
  %2 = tosa.slice %0 {size = array<i64: 1, 3, 12>, start = array<i64: 1, 3, 12>} : (tensor<1x12x24xf32>) -> tensor<1x3x12xf32>
  return %1, %2 : tensor<1x6x12xf32>, tensor<1x3x12xf32>
}

// -----

// CHECK-LABEL: @fold_log_exp
func.func @fold_log_exp(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg{{.*}} : tensor<?x1xf32>
  %0 = tosa.exp %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %1 = tosa.log %0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %1 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @fold_exp_log
func.func @fold_exp_log(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg{{.*}} : tensor<?x1xf32>
  %0 = tosa.log %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %1 = tosa.exp %0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %1 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @fold_negate_negate
func.func @fold_negate_negate(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg{{.*}} : tensor<?x1xf32>
  %0 = tosa.negate %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %1 = tosa.negate %0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %1 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @fold_abs_abs
func.func @fold_abs_abs(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: %[[ABS:.*]] = tosa.abs %arg{{.*}} : (tensor<?x1xf32>) -> tensor<?x1xf32>
  // CHECK: return %[[ABS]] : tensor<?x1xf32>
  %0 = tosa.abs %arg0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  %1 = tosa.abs %0 : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %1 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @fold_reduce_rank_zero
func.func @fold_reduce_rank_zero() {
  // CHECK-NOT: tosa.reduce_min
  // CHECK-NOT: tosa.reverse
  %0 = tensor.empty() : tensor<i32>
  %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<i32>) -> tensor<i32>
  %2 = tosa.reverse %0 {axis = 0 : i32} : (tensor<i32>) -> tensor<i32>
  return
}

// -----

// CHECK-LABEL: @fold_tile_rank_zero
func.func nested @fold_tile_rank_zero() -> tensor<i32> {
  // CHECK-NOT: tosa.tile
  %0 = tensor.empty() : tensor<i32>
  %1 = tosa.tile %0 {multiples = array<i64>} : (tensor<i32>) -> tensor<i32>
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: @reshape_quant_nofold
// check that segfault is fixed
func.func @reshape_quant_nofold() -> tensor<1x1x1x1xi32> {
   %0 = "tosa.const"() {value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = tosa.reshape %0 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<1x1x1x1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %2 = tosa.rescale %1 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x1x1x1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<1x1x1x1xi32>
   return %2 : tensor<1x1x1x1xi32>
}

// -----

// CHECK-LABEL: @add_quant_nofold
// check that segfault is fixed
func.func @add_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   %0 = "tosa.const"() {value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = tosa.add %0, %0 : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @sub_quant_nofold
// check that segfault is fixed
func.func @sub_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   %0 = "tosa.const"() {value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = tosa.sub %0, %0 : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @greater_quant_fold
func.func @greater_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{value = dense<false>
   %2 = "tosa.greater"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @greater_equal_quant_fold
func.func @greater_equal_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{value = dense<true>
   %2 = "tosa.greater_equal"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @equal_quant_fold
func.func @equal_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{value = dense<true>
   %2 = "tosa.equal"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @cast_quant_nofold
func.func @cast_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>> {
  // CHECK: tosa.cast
   %0 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.cast"(%0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>>
}

// -----

// CHECK-LABEL: @reverse_quant_fold
func.func @reverse_quant_fold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: %[[CST:.*]] = "tosa.const"() <{value = dense<0> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: return %[[CST]]
   %0 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.reverse"(%0) { axis = 0 : i32 } : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @select_quant_fold
func.func @select_quant_fold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: %[[CONST_0:.*]] = "tosa.const"() <{value = dense<0> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: return %[[CONST_0]]
   %0 = "tosa.const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
   %1 = "tosa.const"() {value = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %2 = "tosa.const"() {value = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %3 = "tosa.select"(%0, %1, %2) : (tensor<i1>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %3 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @mul_quant_nofold
func.func @mul_quant_nofold() -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: tosa.mul
   %0 = "tosa.const"() {value = dense<0> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.const"() {value = dense<1> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %2 = tosa.mul %0, %1 { shift = 0 : i8} : (tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %2 : tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}


// -----

// CHECK-LABEL: @fold_reciprocal
func.func nested @fold_reciprocal() -> tensor<3x600x1200xf32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{value = dense<8.620690e-03> : tensor<3x600x1200xf32>}> : () -> tensor<3x600x1200xf32>
  // CHECK:           return %[[VAL_0]] : tensor<3x600x1200xf32>
  // CHECK:         }
  %0 = "tosa.const"(){ value = dense<116.0>: tensor<f32> }: () -> tensor<f32>
  %1 = "tosa.cast"(%0) : (tensor<f32>) -> tensor<3x600x1200xf32>
  %2 = "tosa.reciprocal"(%1): (tensor<3x600x1200xf32>) -> tensor<3x600x1200xf32>
  return %2 : tensor<3x600x1200xf32>
}

// -----

// CHECK-LABEL: @do_not_fold_reciprocal_int
func.func nested @do_not_fold_reciprocal_int() -> tensor<3x600x1200xi32> {
  // CHECK:           tosa.reciprocal
  %0 = "tosa.const"(){ value = dense<11>: tensor<i32> }: () -> tensor<i32>
  %1 = "tosa.cast"(%0) : (tensor<i32>) -> tensor<3x600x1200xi32>
  %2 = "tosa.reciprocal"(%1): (tensor<3x600x1200xi32>) -> tensor<3x600x1200xi32>
  return %2 : tensor<3x600x1200xi32>
}

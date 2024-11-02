// RUN: mlir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @argmax_nofold
func.func @argmax_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.argmax"
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @add_zero_different_shape
func.func @add_zero_different_shape(%arg0: tensor<2x3xi32>) -> tensor<4x2x3xi32> {
  // CHECK: tosa.add
  %zeros = "tosa.const"() {value = dense<0> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %1 = "tosa.add"(%arg0, %zeros) : (tensor<2x3xi32>, tensor<4x2x3xi32>) -> tensor<4x2x3xi32>
  return %1 : tensor<4x2x3xi32>
}


// CHECK-LABEL: @add_zero_int
func.func @add_zero_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.add
  %zeros = "tosa.const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = "tosa.add"(%arg0, %zeros) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: @cast_fold
func.func @cast_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.cast"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @cast_nofold
func.func @cast_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xi32> {
  // CHECK: "tosa.cast"
  %0 = "tosa.cast"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xi32>
  return %0 : tensor<?x1xi32>
}

// CHECK-LABEL: @clamp_not_noop
func.func @clamp_not_noop(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: "tosa.clamp"
  %0 = "tosa.clamp"(%arg0) {min_int = 1 : i64, max_int = 4 : i64, min_fp = 1.0 : f32, max_fp = 4.0 : f32} : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @clamp_float_is_noop
func.func @clamp_float_is_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  %0 = "tosa.clamp"(%arg0) {min_int = -128 : i64, max_int = 127 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @clamp_int8_is_noop
func.func @clamp_int8_is_noop(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  %0 = "tosa.clamp"(%arg0) {min_int = -128 : i64, max_int = 127 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// CHECK-LABEL: @clamp_int16_is_noop
func.func @clamp_int16_is_noop(%arg0: tensor<4xi16>) -> tensor<4xi16> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  %0 = "tosa.clamp"(%arg0) {min_int = -32768 : i64, max_int = 32767 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xi16>) -> tensor<4xi16>
  return %0 : tensor<4xi16>
}

// CHECK-LABEL: @clamp_uint8_is_noop
func.func @clamp_uint8_is_noop(%arg0: tensor<4xui8>) -> tensor<4xui8> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  %0 = "tosa.clamp"(%arg0) {min_int = 0 : i64, max_int = 255 : i64, min_fp = -3.40282347E+38 : f32, max_fp = 3.40282347E+38 : f32} :  (tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}

// CHECK-LABEL: @clamp_twice_is_single_clamp
func.func @clamp_twice_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.clamp"(%arg0) {max_fp = 3.000000e+00 : f32, max_int = 2 : i64, min_fp = -3.000000e+00 : f32, min_int = -2 : i64}
  %0 = "tosa.clamp"(%arg0) {max_fp = 3.0 : f32, max_int = 4 : i64, min_fp = -5.0 : f32, min_int = -2 : i64} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = "tosa.clamp"(%0) {max_fp = 5.0 : f32, max_int = 2 : i64, min_fp = -3.0 : f32, min_int = -4 : i64} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// CHECK-LABEL: @concat_fold
func.func @concat_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.concat"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @concat_fold_cast
func.func @concat_fold_cast(%arg0: tensor<?x1xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[VAR0:.*]] = tensor.cast %arg0
  // CHECK: return %[[VAR0]]
  %0 = "tosa.concat"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @conv2d_stride_2
func.func @conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>) -> tensor<4x10x10x3xf32> {
  // CHECK: "tosa.conv2d"
  %weight = "tosa.const"() {value = dense<[[[[1.0, 1.0]]], [[[1.0, 1.0]]], [[[1.0, 1.0]]]]> : tensor<3x1x1x2xf32>} : ()-> tensor<3x1x1x2xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<3xf32>} : ()-> tensor<3xf32>
  %0 = "tosa.conv2d"(%arg0, %weight, %bias) {pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<4x10x10x3xf32>
  return %0 : tensor<4x10x10x3xf32>
}

// CHECK-LABEL: @conv2d_weight_2x2
func.func @conv2d_weight_2x2(%arg0: tensor<4x10x10x1xf32>) -> tensor<4x10x10x1xf32> {
  // CHECK: "tosa.conv2d"
  %weight = "tosa.const"() {value = dense<[[[[1.0], [1.0]], [[1.0], [1.0]]]]> : tensor<1x2x2x1xf32>} : ()-> tensor<1x2x2x1xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : ()-> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %weight, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>) -> tensor<4x10x10x1xf32>
  return %0 : tensor<4x10x10x1xf32>
}

// CHECK-LABEL: @depthwise_conv2d_stride_2
func.func @depthwise_conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: "tosa.depthwise_conv2d"
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// CHECK-LABEL: @depthwise_conv2d_weight_2x2
func.func @depthwise_conv2d_weight_2x2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: "tosa.depthwise_conv2d"
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// CHECK-LABEL: @max_pool2d_is_noop
func.func @max_pool2d_is_noop(%arg0: tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32> {
  // CHECK-NOT: "tosa.max_pool2d"
  // CHECK: return %arg0
  %0 = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32>
  return %0 : tensor<10x1x1x3xf32>
}

// CHECK-LABEL: @pad_noop
func.func @pad_noop(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const"() { value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %0) : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @pad_determine_val_i32
func.func @pad_determine_val_i32(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0> : tensor<i32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @pad_determine_val_f32
func.func @pad_determine_val_f32(%arg0: tensor<?x?xf32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xf32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: @pad_determine_val_quant
func.func @pad_determine_val_quant(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<42> : tensor<i32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) {quantization_info = #tosa.pad_quant<input_zp = 42>} : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: @mul_one_different_shape
func.func @mul_one_different_shape(%arg0: tensor<2x3xf32>) -> tensor<4x2x3xf32> {
  // CHECK: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<4x2x3xf32>} : () -> tensor<4x2x3xf32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
  return %1 : tensor<4x2x3xf32>
}

// CHECK-LABEL: @mul_one_float
func.func @mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// CHECK-LABEL: @mul_one_int
func.func @mul_one_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: @select_same_value
func.func @select_same_value(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "tosa.select"(%arg0, %arg1, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @select_true_value
func.func @select_true_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c1 = "tosa.const"() {value = dense<1> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = "tosa.select"(%c1, %arg0, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @select_false_value
func.func @select_false_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = "tosa.const"() {value = dense<0> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = "tosa.select"(%c0, %arg0, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg1
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @select_not_pred
func.func @select_not_pred(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "tosa.logical_not"(%arg0) : (tensor<2x3xi1>) -> tensor<2x3xi1>
  %1 = "tosa.select"(%0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: "tosa.select"(%arg0, %arg2, %arg1)
  return %1 : tensor<2x3xi32>
}

// CHECK-LABEL: @reduce_all_fold
func.func @reduce_all_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_all"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_all_nofold
func.func @reduce_all_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_all"
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_any_fold
func.func @reduce_any_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_any"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_any_nofold
func.func @reduce_any_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_any"
  %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_max_fold
func.func @reduce_max_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_max"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_max_nofold
func.func @reduce_max_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_max"
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_min_fold
func.func @reduce_min_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_min"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_min_nofold
func.func @reduce_min_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_min"
  %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_prod_fold
func.func @reduce_prod_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_prod"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_prod_nofold
func.func @reduce_prod_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_prod"
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_sum_fold
func.func @reduce_sum_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reduce_sum_nofold
func.func @reduce_sum_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_sum"
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// CHECK-LABEL: @reshape_canonicalize
func.func @reshape_canonicalize(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1, 10]}: (tensor<?x10xf32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// CHECK-LABEL: @reshape_canonicalize_double
func.func @reshape_canonicalize_double(%arg0: tensor<?x10xf32>) -> tensor<?x5xf32> {
  // CHECK: %[[VAR0:.+]] = "tosa.reshape"(%arg0) {new_shape = [-1, 5]}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [5, -1]}: (tensor<?x10xf32>) -> tensor<5x?xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [-1, 5]}: (tensor<5x?xf32>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// CHECK-LABEL: @reshape_canonicalize_const
func.func @reshape_canonicalize_const() -> tensor<1x10xi32> {
  // CHECK: %[[VAR0:.+]] = "tosa.const"() {value = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.const"() {value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 10]} : (tensor<10xi32>) -> tensor<1x10xi32>
  return %1 : tensor<1x10xi32>
}

// CHECK-LABEL: @reshape_canonicalize_const_spat
func.func @reshape_canonicalize_const_spat() -> (tensor<10xi32>, tensor<1x10xi32>) {
  // CHECK-DAG: %[[VAR0:.+]] = "tosa.const"() {value = dense<0> : tensor<10xi32>}
  // CHECK-DAG: %[[VAR1:.+]] = "tosa.const"() {value = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]], %[[VAR1]]
  %0 = "tosa.const"() {value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 10]} : (tensor<10xi32>) -> tensor<1x10xi32>
  return %0 , %1 : tensor<10xi32>, tensor<1x10xi32>
}

// CHECK-LABEL: @reshape_canonicalize_const_sparse
func.func @reshape_canonicalize_const_sparse() -> (tensor<3xi32>, tensor<1x3xi32>) {
  //CHECK: "tosa.reshape"
  %0 = "tosa.const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : ()-> tensor<3xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 3]} : (tensor<3xi32>) -> tensor<1x3xi32>
  return %0 , %1 : tensor<3xi32>, tensor<1x3xi32>
}

// CHECK-LABEL: @slice_fold
func.func @slice_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.slice"(%arg0) { size = [3, 4], start = [0, 0]}: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: @slice_nofold
func.func @slice_nofold(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: "tosa.slice"
  %0 = "tosa.slice"(%arg0) { size = [3, 4], start = [0, 0]}: (tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// CHECK-LABEL: @tile_fold
func.func @tile_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.tile"(%arg0) { multiples = [1, 1] }: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// CHECK-LABEL: @tile_nofold
func.func @tile_nofold(%arg0: tensor<3x4xf32>) -> tensor<3x8xf32> {
  // CHECK: "tosa.tile"
  %0 = "tosa.tile"(%arg0) { multiples = [1, 2] }: (tensor<3x4xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// CHECK-LABEL: @transpose_no_op
func.func @transpose_no_op(%arg0: tensor<3x4x5x6xf32>) -> tensor<3x4x5x6xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.transpose
  %perms = "tosa.const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tosa.transpose"(%arg0, %perms) : (tensor<3x4x5x6xf32>, tensor<4xi32>) -> tensor<3x4x5x6xf32>
  return %1 : tensor<3x4x5x6xf32>
}

// CHECK-LABEL: @single_bit_reshape
// https://github.com/llvm/llvm-project/issues/55440
func.func @single_bit_reshape() -> tensor<1xi1> {
  // CHECK: "tosa.const"() {value = dense<true> : tensor<1xi1>}
  %0 = arith.constant dense<true> : tensor<1x1xi1>
  %1 = "tosa.reshape"(%0) {new_shape = [1]} : (tensor<1x1xi1>) -> tensor<1xi1>
  return %1 : tensor<1xi1>
}

// -----

// CHECK-LABEL: @fold_resize_nearest
func.func @fold_resize_nearest(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %resize = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

// -----

// CHECK-LABEL: @fold_resize_bilinear
func.func @fold_resize_bilinear(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %resize = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [2, 2, 1, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

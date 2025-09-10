// RUN: mlir-opt --split-input-file -canonicalize="test-convergence" %s | FileCheck %s

// CHECK-LABEL: @argmax_nofold
func.func @argmax_nofold(%arg0: tensor<?x1xf32>) -> tensor<1xi32> {
  // CHECK: tosa.argmax
  %0 = tosa.argmax %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----

// CHECK-LABEL: @pad_wh_avg_pool2d_fold
func.func @pad_wh_avg_pool2d_fold(%input: tensor<1x10x8x3xf32>) -> tensor<1x6x5x3xf32> {
  // CHECK-NOT: tosa.pad
  // CHECK: tosa.avg_pool2d
  // CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x11x9x3xf32>
  %pool = tosa.avg_pool2d %padded, %input_zp, %output_zp {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x9x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x6x5x3xf32>
  return %pool : tensor<1x6x5x3xf32>
}

// -----

// CHECK-LABEL: @pad_wh_avg_pool2d_nofold_pad_const
func.func @pad_wh_avg_pool2d_nofold_pad_const(%input: tensor<1x10x8x3xi8>) -> tensor<1x6x5x3xi8> {
  // CHECK: tosa.pad
  // CHECK: tosa.avg_pool2d
  // CHECK-SAME: pad = array<i64: 0, 1, 0, 1>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<15> : tensor<1xi8>}> : ()-> tensor<1xi8>
  %input_zp = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : ()-> tensor<1xi8>
  %output_zp = "tosa.const"() <{values = dense<20> : tensor<1xi8>}> : ()-> tensor<1xi8>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<1x11x9x3xi8>
  %pool = tosa.avg_pool2d %padded, %input_zp, %output_zp {acc_type = i32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x9x3xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x5x3xi8>
  return %pool : tensor<1x6x5x3xi8>
}

// -----

// CHECK-LABEL: @pad_wh_avg_pool2d_nofold_pad_larger_than_kernel
func.func @pad_wh_avg_pool2d_nofold_pad_larger_than_kernel(%input: tensor<1x10x8x3xf32>) -> tensor<1x7x5x3xf32> {
  // CHECK: tosa.pad
  // CHECK: tosa.avg_pool2d
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 3, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %output_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x13x9x3xf32>
  %pool = tosa.avg_pool2d %padded, %input_zp, %output_zp {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x13x9x3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x7x5x3xf32>
  return %pool : tensor<1x7x5x3xf32>
}

// -----

// CHECK-LABEL: @pad_wh_conv2d_fold
func.func @pad_wh_conv2d_fold(%input: tensor<1x8x4x3xf32>, %weight: tensor<1x3x3x3xf32>, %bias: tensor<1xf32>) -> tensor<1x10x8x1xf32> {
  // CHECK-NOT: tosa.pad
  // CHECK: tosa.conv2d
  // CHECK-SAME: pad = array<i64: 2, 2, 3, 3>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 1, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x8x4x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x10x8x3xf32>
  %conv = tosa.conv2d %padded, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<1x10x8x3xf32>, tensor<1x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x10x8x1xf32>
  return %conv : tensor<1x10x8x1xf32>
}

// -----

// CHECK-LABEL: @pad_bwh_conv2d_nofold
func.func @pad_bwh_conv2d_nofold(%input: tensor<1x8x4x3xf32>, %weight: tensor<1x3x3x3xf32>, %bias: tensor<1xf32>) -> tensor<3x10x8x1xf32> {
  // CHECK: tosa.pad
  // CHECK: tosa.conv2d
  // CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
  %pad_shape = tosa.const_shape { values = dense<[1, 1, 1, 1, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x8x4x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<3x10x8x3xf32>
  %conv = tosa.conv2d %padded, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<3x10x8x3xf32>, tensor<1x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<3x10x8x1xf32>
  return %conv : tensor<3x10x8x1xf32>
}

// -----

// CHECK-LABEL: @pad_wh_conv2d_nofold_pad_const
func.func @pad_wh_conv2d_nofold_pad_const(%input: tensor<1x8x4x3xf32>, %weight: tensor<1x3x3x3xf32>, %bias: tensor<1xf32>) -> tensor<1x10x8x1xf32> {
  // CHECK: tosa.pad
  // CHECK: tosa.conv2d
  // CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 1, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<1.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x8x4x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x10x8x3xf32>
  %conv = tosa.conv2d %padded, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<1x10x8x3xf32>, tensor<1x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x10x8x1xf32>
  return %conv : tensor<1x10x8x1xf32>
}

// -----

// CHECK-LABEL: @pad_wh_depthwise_conv2d_fold
func.func @pad_wh_depthwise_conv2d_fold(%input: tensor<1x8x4x3xf32>, %weight: tensor<3x3x3x1xf32>, %bias: tensor<3xf32>) -> tensor<1x10x8x3xf32> {
  // CHECK-NOT: tosa.pad
  // CHECK: tosa.depthwise_conv2d
  // CHECK-SAME: pad = array<i64: 2, 2, 3, 3>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 1, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x8x4x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x10x8x3xf32>
  %conv = tosa.depthwise_conv2d %padded, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<1x10x8x3xf32>, tensor<3x3x3x1xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x10x8x3xf32>
  return %conv : tensor<1x10x8x3xf32>
}

// -----

// CHECK-LABEL: @pad_wh_max_pool2d_fold
func.func @pad_wh_max_pool2d_fold(%input: tensor<1x10x8x3xf32>) -> tensor<1x6x5x3xf32> {
  // CHECK-NOT: tosa.pad
  // CHECK: tosa.max_pool2d
  // CHECK-SAME: pad = array<i64: 1, 1, 1, 1>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<-3.4028235e+38> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x11x9x3xf32>
  %pool = tosa.max_pool2d %padded {kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x9x3xf32>) -> tensor<1x6x5x3xf32>
  return %pool : tensor<1x6x5x3xf32>
}

// -----

// CHECK-LABEL: @pad_wh_max_pool2d_nofold_pad_const
func.func @pad_wh_max_pool2d_nofold_pad_const(%input: tensor<1x10x8x3xf32>) -> tensor<1x6x5x3xf32> {
  // CHECK: tosa.pad
  // CHECK: tosa.max_pool2d
  // CHECK-SAME: pad = array<i64: 0, 1, 0, 1>
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 0, 1, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x11x9x3xf32>
  %pool = tosa.max_pool2d %padded {kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x9x3xf32>) -> tensor<1x6x5x3xf32>
  return %pool : tensor<1x6x5x3xf32>
}

// -----

// CHECK-LABEL: @pad_wh_max_pool2d_no_fold_8k_limit
func.func @pad_wh_max_pool2d_no_fold_8k_limit(%input: tensor<1x10x8x3xf32>) -> tensor<1x6x4101x3xf32> {
  // CHECK: tosa.pad
  // CHECK: tosa.max_pool2d
  %pad_shape = tosa.const_shape { values = dense<[0, 0, 1, 0, 8193, 0, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %pad_const = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %padded = tosa.pad %input, %pad_shape, %pad_const : (tensor<1x10x8x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x11x8201x3xf32>
  %pool = tosa.max_pool2d %padded {kernel = array<i64: 2, 2>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x8201x3xf32>) -> tensor<1x6x4101x3xf32>
  return %pool : tensor<1x6x4101x3xf32>
}

// -----

// CHECK-LABEL: @add_bcast_zero_int
func.func @add_bcast_zero_int(%arg0: tensor<4x2x3xi32>) -> tensor<4x2x3xi32> {
  // CHECK-NOT: tosa.add
  // CHECK: return %arg0
  %zeros = "tosa.const"() {values = dense<0> : tensor<1x1x1xi32>} : () -> tensor<1x1x1xi32>
  %1 = tosa.add %arg0, %zeros : (tensor<4x2x3xi32>, tensor<1x1x1xi32>) -> tensor<4x2x3xi32>
  return %1 : tensor<4x2x3xi32>
}

// -----

// CHECK-LABEL: @add_zero_int
func.func @add_zero_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.add
  %zeros = "tosa.const"() {values = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
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
  %0 = tosa.clamp %arg0 {min_val = 1 : i32, max_val = 4 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: @clamp_f32_not_noop
func.func @clamp_f32_not_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = -3.40282347E+38 : f32, max_val = 3.40282347E+38 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_f16_is_noop
func.func @clamp_f16_is_noop(%arg0: tensor<4xf16>) -> tensor<4xf16> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  // 0x7C00 and 0xFC00 are respectively positive and negative F32 infinity.
  %0 = tosa.clamp %arg0 {max_val = 0x7C00 : f16, min_val = 0xFC00 : f16} : (tensor<4xf16>) -> tensor<4xf16>
  return %0 : tensor<4xf16>
}

// -----

// CHECK-LABEL: @clamp_f32_is_noop
func.func @clamp_f32_is_noop(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: "tosa.clamp"
  // 0xFF800000 and 0x7F800000 are respectively negative and positive F32 infinity.
  %0 = tosa.clamp %arg0 {min_val = 0xFF800000 : f32, max_val = 0x7F800000 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_boolean_is_noop
func.func @clamp_boolean_is_noop(%arg0: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = false, max_val = true} : (tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: @clamp_boolean_dynamic_is_noop
func.func @clamp_boolean_dynamic_is_noop(%arg0: tensor<?xi1>) -> tensor<?xi1> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = false, max_val = true} : (tensor<?xi1>) -> tensor<?xi1>
  return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: @clamp_int8_is_noop
func.func @clamp_int8_is_noop(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = -128 : i8, max_val = 127 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @clamp_int16_is_noop
func.func @clamp_int16_is_noop(%arg0: tensor<4xi16>) -> tensor<4xi16> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = -32768 : i16, max_val = 32767 : i16} :  (tensor<4xi16>) -> tensor<4xi16>
  return %0 : tensor<4xi16>
}

// -----

// CHECK-LABEL: @clamp_uint8_is_noop
func.func @clamp_uint8_is_noop(%arg0: tensor<4xui8>) -> tensor<4xui8> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.clamp
  %0 = tosa.clamp %arg0 {min_val = 0 : ui8, max_val = 255 : ui8} :  (tensor<4xui8>) -> tensor<4xui8>
  return %0 : tensor<4xui8>
}

// -----

// CHECK-LABEL: @clamp_twice_is_single_clamp
func.func @clamp_twice_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_val = 2 : i8, min_val = -2 : i8}
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK: @disjoint_clamp_twice_is_not_single_clamp(%[[INPUT:.*]]: tensor<4xi8>)
func.func @disjoint_clamp_twice_is_not_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: %[[CLAMP_1:.*]] = tosa.clamp %[[INPUT]] {max_val = -5 : i8, min_val = -10 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  // CHECK-NEXT: tosa.clamp %[[CLAMP_1]] {max_val = 5 : i8, min_val = 1 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.clamp %arg0 {max_val = -5 : i8, min_val = -10 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 5 : i8, min_val = 1 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @clamp_twice_with_nan_propagate_is_single_clamp
func.func @clamp_twice_with_nan_propagate_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_val = 2 : i8, min_val = -2 : i8}
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8, nan_mode = PROPAGATE} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8, nan_mode = PROPAGATE} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @clamp_twice_with_nan_ignore_is_single_clamp
func.func @clamp_twice_with_nan_ignore_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_val = 2 : i8, min_val = -2 : i8, nan_mode = IGNORE}
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8, nan_mode = IGNORE} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8, nan_mode = IGNORE} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: @clamp_twice_with_nan_ignore_propagate_is_single_clamp
func.func @clamp_twice_with_nan_ignore_propagate_is_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.clamp %arg0 {max_val = 2 : i8, min_val = -2 : i8, nan_mode = IGNORE}
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8, nan_mode = IGNORE} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8, nan_mode = PROPAGATE} :  (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK: @clamp_twice_with_nan_propagate_ignore_is_not_single_clamp(%[[INPUT:.*]]: tensor<4xi8>)
func.func @clamp_twice_with_nan_propagate_ignore_is_not_single_clamp(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: %[[CLAMP_1:.*]] = tosa.clamp %[[INPUT]] {max_val = 4 : i8, min_val = -2 : i8} :  (tensor<4xi8>) -> tensor<4xi8>
  // CHECK-NEXT: tosa.clamp %[[CLAMP_1]] {max_val = 2 : i8, min_val = -4 : i8, nan_mode = IGNORE} :  (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.clamp %arg0 {max_val = 4 : i8, min_val = -2 : i8, nan_mode = PROPAGATE} :  (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.clamp %0 {max_val = 2 : i8, min_val = -4 : i8, nan_mode = IGNORE} :  (tensor<4xi8>) -> tensor<4xi8>
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
func.func @conv2d_stride_2(%arg0: tensor<4x11x11x2xf32>) -> tensor<4x6x6x3xf32> {
  // CHECK: tosa.conv2d
  %weight = "tosa.const"() {values = dense<[[[[1.0, 1.0]]], [[[1.0, 1.0]]], [[[1.0, 1.0]]]]> : tensor<3x1x1x2xf32>} : ()-> tensor<3x1x1x2xf32>
  %bias = "tosa.const"() {values = dense<0.0> : tensor<3xf32>} : ()-> tensor<3xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %0 = tosa.conv2d %arg0, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, dilation = array<i64: 1, 1>} : (tensor<4x11x11x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x6x6x3xf32>
  return %0 : tensor<4x6x6x3xf32>
}

// -----

// CHECK-LABEL: @conv2d_weight_2x2
func.func @conv2d_weight_2x2(%arg0: tensor<4x10x10x1xf32>) -> tensor<4x9x9x1xf32> {
  // CHECK: tosa.conv2d
  %weight = "tosa.const"() {values = dense<[[[[1.0], [1.0]], [[1.0], [1.0]]]]> : tensor<1x2x2x1xf32>} : ()-> tensor<1x2x2x1xf32>
  %bias = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : ()-> tensor<1xf32>
  %input_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %weight_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : ()-> tensor<1xf32>
  %0 = tosa.conv2d %arg0, %weight, %bias, %input_zp, %weight_zp {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x9x9x1xf32>
  return %0 : tensor<4x9x9x1xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_stride_2
func.func @depthwise_conv2d_stride_2(%arg0: tensor<4x11x11x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<4x6x6x6xf32> {
  // CHECK: tosa.depthwise_conv2d
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, dilation = array<i64: 1, 1>} : (tensor<4x11x11x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x6x6x6xf32>
  return %0 : tensor<4x6x6x6xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_weight_2x2
func.func @depthwise_conv2d_weight_2x2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<4x9x9x6xf32> {
  // CHECK: tosa.depthwise_conv2d
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2, %arg3, %arg4 {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<4x10x10x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4x9x9x6xf32>
  return %0 : tensor<4x9x9x6xf32>
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
  %0 = tosa.const_shape { values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = tosa.pad %arg0, %0, %pad_const : (tensor<?x?xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_noop_padding_mismatch_nofold
func.func @pad_noop_padding_mismatch_nofold(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[PAD:.+]] = tosa.pad
  // CHECK: return %[[PAD]]
  %shape = tosa.const_shape { values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %1 = tosa.pad %arg0, %shape, %pad_const : (tensor<?x?xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_noop_type_mismatch_nofold
func.func @pad_noop_type_mismatch_nofold(%arg0: tensor<10xf32>) -> tensor<?xf32> {
  // CHECK: %[[PAD:.+]] = tosa.pad
  // CHECK: return %[[PAD]]
  %shape = tosa.const_shape { values = dense<[1, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = tosa.pad %arg0, %shape, %pad_const : (tensor<10xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_i32
func.func @pad_determine_val_i32(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}
  // CHECK-DAG: %[[PADDING:.+]] = tosa.const_shape {values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: tosa.pad %arg0, %[[PADDING]], %[[ZERO]]
  %pad_const = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = tosa.const_shape { values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.pad %arg0, %0, %pad_const : (tensor<?x?xi32>, !tosa.shape<4>, tensor<1xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @pad_determine_val_f32
func.func @pad_determine_val_f32(%arg0: tensor<?x?xf32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{values = dense<3.140000e+00> : tensor<1xf32>}
  // CHECK-DAG: %[[PADDING:.+]] = tosa.const_shape {values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: tosa.pad %arg0, %[[PADDING]], %[[ZERO]]
  %pad_const = "tosa.const"() {values = dense<3.14> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = tosa.const_shape { values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.pad %arg0, %0, %pad_const : (tensor<?x?xf32>, !tosa.shape<4>, tensor<1xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_quant
func.func @pad_determine_val_quant(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG: %[[ZERO:.+]] = "tosa.const"() <{values = dense<3> : tensor<1xi32>}
  // CHECK-DAG: %[[PADDING:.+]] = tosa.const_shape {values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: tosa.pad %arg0, %[[PADDING]], %[[ZERO]]
  %pad_const = "tosa.const"() {values =dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = tosa.const_shape { values = dense<[1, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %1 = tosa.pad %arg0, %0, %pad_const {input_zp = 42 : i32} : (tensor<?x?xi32>, !tosa.shape<4>, tensor<1xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @mul_one_float
func.func @mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %ones = "tosa.const"() {values = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_bcast_one_float
func.func @mul_bcast_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {values = dense<1.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %ones, %arg0, %shift : (tensor<1x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_one_int
func.func @mul_one_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %ones = "tosa.const"() {values = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @mul_one_int_and_shift
func.func @mul_one_int_and_shift(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<1> : tensor<2x3xi32>}>
  // CHECK-DAG: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<31> : tensor<1xi8>}>
  // CHECK: %[[VAL_3:.*]] = tosa.mul %arg0, %[[VAL_1]], %[[VAL_2]] : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>)
  %ones = "tosa.const"() {values = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %shift = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %arg0, %ones, %shift : (tensor<2x3xi32>, tensor<2x3xi32>, tensor<1xi8>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @mul_zero_broadcast
func.func @mul_zero_broadcast(%arg0: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  // CHECK: %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<2x3xf32>}
  // CHECK-NOT: tosa.mul
  %zeros = "tosa.const"() {values = dense<0.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %1 = tosa.mul %arg0, %zeros, %shift : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<2x3xf32>

  // CHECK-NOT: tosa.mul
  // CHECK: return %[[ZERO]], %[[ZERO]]
  %2 = tosa.mul %zeros, %arg0, %shift : (tensor<1x1xf32>, tensor<2x3xf32>, tensor<1xi8>) -> tensor<2x3xf32>
  return %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_zero_dynamic_nofold
// CHECK-SAME:                    %[[ARG0:.*]]: tensor<?x17xf32>) -> tensor<?x17xf32> {
// CHECK:           %[[ZERO:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
// CHECK:           %[[SHIFT:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[ZERO]], %[[SHIFT]] : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
// CHECK:           return %[[MUL]]
func.func @mul_zero_dynamic_nofold(%arg0: tensor<?x17xf32>) -> tensor<?x17xf32> {
  %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
  return %2 : tensor<?x17xf32>
}

// -----

// CHECK-LABEL: @mul_one_dynamic_fold
// CHECK-SAME:                    %[[ARG0:.*]]: tensor<?x17xf32>) -> tensor<?x17xf32> {
// CHECK:           return %[[ARG0]]
func.func @mul_one_dynamic_fold(%arg0: tensor<?x17xf32>) -> tensor<?x17xf32> {
  %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
  %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %2 = tosa.mul %arg0, %0, %1 : (tensor<?x17xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<?x17xf32>
  return %2 : tensor<?x17xf32>
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
  %c1 = "tosa.const"() {values = dense<1> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
  %0 = tosa.select %c1, %arg0, %arg1 : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  // CHECK: return %arg0
  // CHECK-NOT: tosa.select
  return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @select_false_value
func.func @select_false_value(%arg0: tensor<2x3xi32>, %arg1: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %c0 = "tosa.const"() {values = dense<0> : tensor<2x3xi1>} : () -> tensor<2x3xi1>
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

// CHECK-LABEL: @reduce_product_fold
func.func @reduce_product_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = tosa.reduce_product %arg0 {axis = 1 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_product_nofold
func.func @reduce_product_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: tosa.reduce_product
  %0 = tosa.reduce_product %arg0 {axis = 0 : i32}: (tensor<?x1xf32>) -> tensor<?x1xf32>
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
  %0 = "tosa.const_shape"() {values = dense<[-1, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %arg0, %0 : (tensor<?x10xf32>, !tosa.shape<2>) -> tensor<?x10xf32>
  return %1 : tensor<?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_dyn_nofold
func.func @reshape_canonicalize_dyn_nofold(%arg0: tensor<?x?x10xf32>) -> tensor<?x?x10xf32> {
  // CHECK: %[[SHAPE:.+]] = tosa.const_shape {values = dense<[-1, 2, 10]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // CHECK: %[[VAR0:.+]] = tosa.reshape %arg0, %[[SHAPE]] : (tensor<?x?x10xf32>, !tosa.shape<3>) -> tensor<?x?x10xf32>
  // CHECK: return %[[VAR0]] : tensor<?x?x10xf32>
  %s = "tosa.const_shape"() {values = dense<[-1, 2, 10]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %0 = tosa.reshape %arg0, %s : (tensor<?x?x10xf32>, !tosa.shape<3>) -> tensor<?x?x10xf32>
  return %0 : tensor<?x?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_double
func.func @reshape_canonicalize_double(%arg0: tensor<?x10xf32>) -> tensor<?x5xf32> {
  // CHECK: %[[VAL_0:.*]] = tosa.const_shape {values = dense<[-1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: %[[VAL_1:.*]] = tosa.reshape %arg0, %[[VAL_0]]
  // CHECK: return %[[VAL_1]]
  %cst0 = "tosa.const_shape"() <{values = dense<[5, -1]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %0 = tosa.reshape %arg0, %cst0 : (tensor<?x10xf32>, !tosa.shape<2>) -> tensor<5x?xf32>
  %cst1 = "tosa.const_shape"() <{values = dense<[-1, 5]> : tensor<2xindex>}> : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %cst1 : (tensor<5x?xf32>, !tosa.shape<2>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const
func.func @reshape_canonicalize_const() -> tensor<1x5xi32> {
  // CHECK: %[[VAR0:.+]] = "tosa.const"() <{values = dense<{{\[\[}}0, 1, 2, 3, 4]]> : tensor<1x5xi32>}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.const"() {values = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %1 = "tosa.const_shape"() {values = dense<[1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %2 = tosa.reshape %0, %1 : (tensor<5xi32>, !tosa.shape<2>) -> tensor<1x5xi32>
  return %2 : tensor<1x5xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_dynamic
func.func @reshape_canonicalize_const_dynamic() -> tensor<1x?xi32> {
  // CHECK: tosa.reshape
  %0 = "tosa.const"() {values = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  %2 = "tosa.const_shape"() {values = dense<[1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<5xi32>, !tosa.shape<2>) -> tensor<1x?xi32>
  return %1 : tensor<1x?xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_splat
func.func @reshape_canonicalize_const_splat() -> (tensor<10xi32>, tensor<1x10xi32>) {
  // CHECK-DAG: %[[VAR0:.+]] = "tosa.const"() <{values = dense<0> : tensor<10xi32>}
  // CHECK-DAG: %[[VAR1:.+]] = "tosa.const"() <{values = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]], %[[VAR1]]
  %0 = "tosa.const"() {values = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %2 = "tosa.const_shape"() {values = dense<[1, 10]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<10xi32>, !tosa.shape<2>) -> tensor<1x10xi32>
  return %0 , %1 : tensor<10xi32>, tensor<1x10xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_sparse
func.func @reshape_canonicalize_const_sparse() -> (tensor<3xi32>, tensor<1x3xi32>) {
  // CHECK: tosa.reshape
  %0 = "tosa.const"() {values = dense<[1, 2, 3]> : tensor<3xi32>} : ()-> tensor<3xi32>
  %2 = "tosa.const_shape"() {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<3xi32>, !tosa.shape<2>) -> tensor<1x3xi32>
  return %0 , %1 : tensor<3xi32>, tensor<1x3xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_quant_nofold
func.func @reshape_canonicalize_quant_nofold() -> (tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>) {
  // disabled folding for quantized element types
  // CHECK{LITERAL}: "tosa.const"() <{values = dense<[1, 2, 3]> : tensor<3xi8>}> : () -> tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>
  // CHECK{LITERAL}: tosa.reshape %0, %1 : (tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>, !tosa.shape<2>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %0 = "tosa.const"() {values = dense<[1, 2, 3]> : tensor<3xi8>} : ()-> tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>
  %2 = "tosa.const_shape"() {values = dense<[1, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.reshape %0, %2 : (tensor<3x!quant.uniform<i8:f32, 1.000000e+00>>, !tosa.shape<2>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  return %1 :  tensor<1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: @transpose_canonicalize_strip_quant
func.func @transpose_canonicalize_strip_quant() -> (tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>) {
  // CHECK-DAG: %[[SHAPE:.*]] = tosa.const_shape {values = dense<[2, 1, 3]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // CHECK-DAG: %[[CONST:.*]] = "tosa.const"() <{values = dense<0> : tensor<1x2x3xi8>}> : () -> tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>
  // CHECK: tosa.reshape %[[CONST]], %[[SHAPE]] : (tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>, !tosa.shape<3>) -> tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %0 = "tosa.const"() {values = dense<0> : tensor<1x2x3xi8>} : ()-> tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>
  %1 = tosa.transpose %0 { perms = array<i32: 1, 0, 2> }: (tensor<1x2x3x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
  return %1 :  tensor<2x1x3x!quant.uniform<i8:f32, 1.000000e+00>>
}

// -----

// CHECK-LABEL: @slice_fold
func.func @slice_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = tosa.const_shape {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.const_shape {values = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: return %arg0
  %3 = tosa.slice %arg0, %0, %1 : (tensor<3x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x4xf32>
  return %3 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @slice_nofold
func.func @slice_nofold(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = tosa.const_shape {values = dense<[0, 0]> : tensor<2xindex>} : () -> !tosa.shape<2>
  %1 = tosa.const_shape {values = dense<[3, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
  // CHECK: tosa.slice
  %3 = tosa.slice %arg0, %0, %1 : (tensor<?x4xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tile_fold
func.func @tile_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %cst = tosa.const_shape { values = dense<1> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %cst: (tensor<3x4xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @tile_nofold
func.func @tile_nofold(%arg0: tensor<3x4xf32>) -> tensor<3x8xf32> {
  // CHECK: tosa.tile
  %cst = tosa.const_shape { values = dense<[1, 2]> : tensor<2xindex> } : () -> !tosa.shape<2>
  %0 = tosa.tile %arg0, %cst: (tensor<3x4xf32>, !tosa.shape<2>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// -----

// CHECK-LABEL: @transpose_no_op
func.func @transpose_no_op(%arg0: tensor<3x4x5x6xf32>) -> tensor<3x4x5x6xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.transpose
  %1 = tosa.transpose %arg0 { perms = array<i32: 0, 1, 2, 3> }: (tensor<3x4x5x6xf32>) -> tensor<3x4x5x6xf32>
  return %1 : tensor<3x4x5x6xf32>
}

// -----

// CHECK-LABEL: @transpose_is_reshape
func.func @transpose_is_reshape(%arg0: tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32> {
  // CHECK: %[[CONST0:.+]] = tosa.const_shape {values = dense<[1, 4, 1, 5]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: tosa.reshape %arg0, %[[CONST0]]
  %0 = tosa.transpose %arg0 { perms = array<i32: 3, 1, 0, 2> }: (tensor<1x4x5x1xf32>) -> tensor<1x4x1x5xf32>
  return %0 : tensor<1x4x1x5xf32>
}

// -----

// CHECK-LABEL: @transpose_is_reshape_unknown_dim
func.func @transpose_is_reshape_unknown_dim(%arg0: tensor<1x4x?x1xf32>) -> tensor<1x4x1x?xf32> {
  // CHECK: %[[CONST0:.+]] = tosa.const_shape {values = dense<[1, 4, 1, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  // CHECK: tosa.reshape %arg0, %[[CONST0]]
  %0 = tosa.transpose %arg0 { perms = array<i32: 3, 1, 0, 2> }: (tensor<1x4x?x1xf32>) -> tensor<1x4x1x?xf32>
  return %0 : tensor<1x4x1x?xf32>
}

// -----

// CHECK-LABEL: @single_bit_reshape
// https://github.com/llvm/llvm-project/issues/55440
func.func @single_bit_reshape() -> tensor<1xi1> {
  // CHECK: "tosa.const"() <{values = dense<true> : tensor<1xi1>}
  %0 = arith.constant dense<true> : tensor<1x1xi1>
  %2 = "tosa.const_shape"() <{values = dense<1> : tensor<1xindex>}> : () -> !tosa.shape<1>
  %1 = tosa.reshape %0, %2 : (tensor<1x1xi1>, !tosa.shape<1>) -> tensor<1xi1>
  return %1 : tensor<1xi1>
}

// -----

// CHECK-LABEL: @fold_resize_nearest
func.func @fold_resize_nearest(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %scale = tosa.const_shape { values = dense<[2, 2, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %resize = tosa.resize %arg0, %scale, %offset, %border {mode = NEAREST_NEIGHBOR} : (tensor<1x15x13x1xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

// -----

// CHECK-LABEL: @fold_resize_bilinear
func.func @fold_resize_bilinear(%arg0 : tensor<1x15x13x1xi8>) -> tensor<1x15x13x1xi8> {
  // CHECK: return %arg0
  %scale = tosa.const_shape { values = dense<[2, 2, 1, 1]> : tensor<4xindex> } : () -> !tosa.shape<4>
  %offset = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %border = tosa.const_shape { values = dense<0> : tensor<2xindex> } : () -> !tosa.shape<2>
  %resize = tosa.resize %arg0, %scale, %offset, %border {mode = NEAREST_NEIGHBOR} : (tensor<1x15x13x1xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x15x13x1xi8>
  return %resize : tensor<1x15x13x1xi8>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_final_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12x1xf32>, %[[VAL_1:.*]]: tensor<1x12x12x1xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]] : tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>
func.func @canonicalize_concat_slice_final_axis(%arg0 : tensor<1x12x12x1xf32>, %arg1 : tensor<1x12x12x1xf32>) -> (tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 3 : i32} : (tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>) -> tensor<1x12x12x2xf32>
  %1 = tosa.const_shape {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %2 = tosa.const_shape {values = dense<[0, 0, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %3 = tosa.const_shape {values = dense<[1, 12, 12, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %4 = tosa.slice %0, %1, %3 : (tensor<1x12x12x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x12x12x1xf32>
  %5 = tosa.slice %0, %2, %3 : (tensor<1x12x12x2xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x12x12x1xf32>
  return %4, %5 : tensor<1x12x12x1xf32>, tensor<1x12x12x1xf32>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_middle_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]] : tensor<1x12x12xf32>, tensor<1x12x12xf32>
func.func @canonicalize_concat_slice_middle_axis(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x12x12xf32>, tensor<1x12x12xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x24x12xf32>
  %1 = tosa.const_shape {values = dense<[0, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.const_shape {values = dense<[0, 12, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape {values = dense<[1, 12, 12]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.slice %0, %1, %3 : (tensor<1x24x12xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x12x12xf32>
  %5 = tosa.slice %0, %2, %3 : (tensor<1x24x12xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x12x12xf32>
  return %4, %5 : tensor<1x12x12xf32>, tensor<1x12x12xf32>
}

// -----

// CHECK-LABEL: @canonicalize_cross_concat_inputs
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[1, 12, 20]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 12, 15]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<[0, 0, 4]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK: %[[VAL_6:.*]] = tosa.concat %[[VAL_0]], %[[VAL_1]] {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
// CHECK: %[[VAL_7:.*]] = tosa.slice %[[VAL_6]], %[[VAL_5]], %[[VAL_3]]
// CHECK: %[[VAL_8:.*]] = tosa.slice %[[VAL_6]], %[[VAL_4]], %[[VAL_2]]
// CHECK: return %[[VAL_7]], %[[VAL_8]] : tensor<1x12x15xf32>, tensor<1x12x20xf32>
func.func @canonicalize_cross_concat_inputs(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x12x15xf32>, tensor<1x12x20xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
  %1 = tosa.const_shape {values = dense<[0, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.const_shape {values = dense<[0, 0, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape {values = dense<[1, 12, 15]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape {values = dense<[1, 12, 20]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = tosa.slice %0, %1, %3 : (tensor<1x12x24xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x12x15xf32>
  %6 = tosa.slice %0, %2, %4 : (tensor<1x12x24xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x12x20xf32>
  return %5, %6 : tensor<1x12x15xf32>, tensor<1x12x20xf32>
}

// -----

// CHECK-LABEL: @canonicalize_concat_slice_on_non_concat_axis
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x12x12xf32>, %[[VAL_1:.*]]: tensor<1x12x12xf32>
// CHECK-DAG: %[[VAL_2:.*]] = tosa.const_shape  {values = dense<[1, 3, 0]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_3:.*]] = tosa.const_shape  {values = dense<[1, 3, 12]> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_4:.*]] = tosa.const_shape  {values = dense<0> : tensor<3xindex>}
// CHECK-DAG: %[[VAL_5:.*]] = tosa.const_shape  {values = dense<[1, 6, 12]> : tensor<3xindex>}
// CHECK: %[[VAL_6:.*]] = tosa.slice %[[VAL_0]], %[[VAL_4]], %[[VAL_5]]
// CHECK: %[[VAL_7:.*]] = tosa.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_3]]
// CHECK: return %[[VAL_6]], %[[VAL_7]] : tensor<1x6x12xf32>, tensor<1x3x12xf32>
func.func @canonicalize_concat_slice_on_non_concat_axis(%arg0 : tensor<1x12x12xf32>, %arg1 : tensor<1x12x12xf32>) -> (tensor<1x6x12xf32>, tensor<1x3x12xf32>) {
  %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<1x12x12xf32>, tensor<1x12x12xf32>) -> tensor<1x12x24xf32>
  %1 = tosa.const_shape {values = dense<[0, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2 = tosa.const_shape {values = dense<[1, 6, 12]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape {values = dense<[1, 3, 12]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.slice %0, %1, %2 : (tensor<1x12x24xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x6x12xf32>
  %5 = tosa.slice %0, %3, %3 : (tensor<1x12x24xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x3x12xf32>
  return %4, %5 : tensor<1x6x12xf32>, tensor<1x3x12xf32>
}

// -----

// CHECK-LABEL: @canonicalize_pad_slice_overlap
// CHECK-DAG: %[[PAD_CONST:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[ZERO:.*]] = tosa.const_shape {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[PADDING:.*]] = tosa.const_shape {values = dense<[0, 0, 0, 0, 1, 1, 0, 0]> : tensor<8xindex>}
// CHECK-DAG: %[[SLICE_SIZE:.*]] = tosa.const_shape  {values = dense<[-1, 14, 18, 3]> : tensor<4xindex>}
// CHECK: %[[PADDED:.*]] = tosa.pad %arg0, %[[PADDING]], %[[PAD_CONST]]
// CHECK: %[[SLICED:.*]] = tosa.slice %[[PADDED]], %[[ZERO]], %[[SLICE_SIZE]]
func.func @canonicalize_pad_slice_overlap(%arg0: tensor<?x16x16x3xf32>) -> tensor<?x14x18x3xf32> {
  %pad_const = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %padding = tosa.const_shape  {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %padded = tosa.pad %arg0, %padding, %pad_const : (tensor<?x16x16x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<?x16x20x3xf32>
  %start = tosa.const_shape  {values = dense<[0, 0, 1, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape  {values = dense<[-1, 14, 18, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %sliced = tosa.slice %padded, %start, %size : (tensor<?x16x20x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x14x18x3xf32>
  return %sliced : tensor<?x14x18x3xf32>
}

// -----

// CHECK-LABEL: @canonicalize_pad_slice_inside
// CHECK-DAG: %[[SLICE_START:.*]] = tosa.const_shape  {values = dense<[0, 1, 2, 0]> : tensor<4xindex>}
// CHECK-DAG: %[[SLICE_SIZE:.*]] = tosa.const_shape  {values = dense<[1, 14, 10, 3]> : tensor<4xindex>}
// CHECK-NOT: tosa.pad
// CHECK: %[[SLICED:.*]] = tosa.slice %arg0, %[[SLICE_START]], %[[SLICE_SIZE]]
func.func @canonicalize_pad_slice_inside(%arg0: tensor<1x16x16x3xf32>) -> tensor<1x14x14x3xf32> {
  %pad_const = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %padding = tosa.const_shape  {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %padded = tosa.pad %arg0, %padding, %pad_const : (tensor<1x16x16x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x16x20x3xf32>
  %start = tosa.const_shape  {values = dense<[0, 1, 4, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape  {values = dense<[1, 14, 10, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %sliced = tosa.slice %padded, %start, %size : (tensor<1x16x20x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x14x14x3xf32>
  return %sliced : tensor<1x14x14x3xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_pad_slice_exact
// CHECK-DAG: %[[PAD_CONST:.*]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG: %[[ZERO:.*]] = tosa.const_shape {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-DAG: %[[PADDING:.*]] = tosa.const_shape {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>}
// CHECK-DAG: %[[SLICE_SIZE:.*]] = tosa.const_shape  {values = dense<[1, 16, 20, 2]> : tensor<4xindex>}
// CHECK: %[[PADDED:.*]] = tosa.pad %arg0, %[[PADDING]], %[[PAD_CONST]]
// CHECK: %[[SLICED:.*]] = tosa.slice %[[PADDED]], %[[ZERO]], %[[SLICE_SIZE]]
func.func @canonicalize_pad_slice_exact(%arg0: tensor<1x16x16x3xf32>) -> tensor<1x16x20x2xf32> {
  %pad_const = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %padding = tosa.const_shape  {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %padded = tosa.pad %arg0, %padding, %pad_const : (tensor<1x16x16x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x16x20x3xf32>
  %start = tosa.const_shape  {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape  {values = dense<[1, 16, 20, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %sliced = tosa.slice %padded, %start, %size : (tensor<1x16x20x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x16x20x2xf32>
  return %sliced : tensor<1x16x20x2xf32>
}

// -----

// CHECK-LABEL: func @canonicalize_pad_slice_dynamic_noupdate
// CHECK-DAG: tosa.const_shape {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>}
// CHECK-DAG: tosa.const_shape {values = dense<[1, 16, 15, 2]> : tensor<4xindex>}
// CHECK: tosa.pad
// CHECK: tosa.slice
func.func @canonicalize_pad_slice_dynamic_noupdate(%arg0: tensor<1x16x?x3xf32>) -> tensor<1x16x?x2xf32> {
  %pad_const = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  %padding = tosa.const_shape  {values = dense<[0, 0, 0, 0, 2, 2, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
  %padded = tosa.pad %arg0, %padding, %pad_const : (tensor<1x16x?x3xf32>, !tosa.shape<8>, tensor<1xf32>) -> tensor<1x16x?x3xf32>
  %start = tosa.const_shape  {values = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape  {values = dense<[1, 16, 15, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %sliced = tosa.slice %padded, %start, %size : (tensor<1x16x?x3xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<1x16x?x2xf32>
  return %sliced : tensor<1x16x?x2xf32>
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
  %in_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %out_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.negate %arg0, %in_zp, %out_zp : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  %1 = tosa.negate %0, %in_zp, %out_zp : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  return %1 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @no_fold_negate_negate_non_const_zp
func.func @no_fold_negate_negate_non_const_zp(%arg0: tensor<?x1xf32>, %in_zp: tensor<1xf32>) -> tensor<?x1xf32> {
  // cannot fold if any zp is not constant
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  %zero = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.negate %arg0, %in_zp, %zero : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  %1 = tosa.negate %0, %zero, %zero : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  %2 = tosa.negate %1, %zero, %in_zp : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  %3 = tosa.negate %2, %zero, %zero : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  %4 = tosa.negate %3, %in_zp, %zero : (tensor<?x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x1xf32>
  return %4 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @no_fold_negate_negate_non_zero_zp
func.func @no_fold_negate_negate_non_zero_zp(%arg0: tensor<?x1xi8>) -> tensor<?x1xi8> {
  // cannot fold if any zp is not constant 0
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  // CHECK: tosa.negate
  %zero = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %one = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
  %0 = tosa.negate %arg0, %zero, %one : (tensor<?x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1xi8>
  %1 = tosa.negate %0, %zero, %zero : (tensor<?x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1xi8>
  %2 = tosa.negate %1, %one, %zero : (tensor<?x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1xi8>
  %3 = tosa.negate %2, %zero, %zero : (tensor<?x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1xi8>
  %4 = tosa.negate %3, %zero, %one : (tensor<?x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1xi8>
  return %4 : tensor<?x1xi8>
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

// CHECK-LABEL: @reshape_quant_nofold
// check that segfault is fixed
func.func @reshape_quant_nofold() -> tensor<1x1x1x1xi32> {
   %0 = "tosa.const"() {values = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %cst0 = "tosa.const_shape"() {values = dense<[1, 1, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
   %1 = tosa.reshape %0, %cst0 : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, !tosa.shape<4>) -> tensor<1x1x1x1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32> } : () -> tensor<1xi32>
   %shift = "tosa.const"() {values = dense<30> : tensor<1xi8> } : () -> tensor<1xi8>
   %input_zp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
   %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
   %2 = tosa.rescale %1, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = DOUBLE_ROUND, scale32 = true, per_channel = false, input_unsigned = false, output_unsigned = false} : (tensor<1x1x1x1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1x1xi32>
   return %2 : tensor<1x1x1x1xi32>
}

// -----

// CHECK-LABEL: @add_quant_nofold
// check that segfault is fixed
func.func @add_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   %0 = "tosa.const"() {values = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = tosa.add %0, %0 : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @sub_quant_nofold
// check that segfault is fixed
func.func @sub_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   %0 = "tosa.const"() {values = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = tosa.sub %0, %0 : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @greater_quant_fold
func.func @greater_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {values = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{values = dense<false>
   %2 = "tosa.greater"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @greater_equal_quant_fold
func.func @greater_equal_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {values = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{values = dense<true>
   %2 = "tosa.greater_equal"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @equal_quant_fold
func.func @equal_quant_fold() -> tensor<i1> {
   %0 = "tosa.const"() {values = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: "tosa.const"() <{values = dense<true>
   %2 = "tosa.equal"(%0, %0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<i1>
   return %2 : tensor<i1>
}

// -----

// CHECK-LABEL: @cast_quant_nofold
func.func @cast_quant_nofold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>> {
  // CHECK: tosa.cast
   %0 = "tosa.const"() {values = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.cast"(%0) : (tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>>
   return %1 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:3>>
}

// -----

// CHECK-LABEL: @reverse_quant_fold
func.func @reverse_quant_fold() -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: %[[CST:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: return %[[CST]]
   %0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.reverse"(%0) { axis = 0 : i32 } : (tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %1 : tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @select_quant_fold
func.func @select_quant_fold() -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: %[[CONST_0:.*]] = "tosa.const"() <{values = dense<0> : tensor<i8>}> : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   // CHECK: return %[[CONST_0]]
   %0 = "tosa.const"() {values = dense<true> : tensor<i1>} : () -> tensor<i1>
   %1 = "tosa.const"() {values = dense<0> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %2 = "tosa.const"() {values = dense<127> : tensor<i8>} : () -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %3 = "tosa.select"(%0, %1, %2) : (tensor<i1>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>) -> tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %3 : tensor<!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}

// -----

// CHECK-LABEL: @mul_quant_nofold
func.func @mul_quant_nofold() -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>> {
   // CHECK: tosa.mul
   %0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %1 = "tosa.const"() {values = dense<1> : tensor<1xi8>} : () -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
   %2 = tosa.mul %0, %1, %shift : (tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>, tensor<1xi8>) -> tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
   return %2 : tensor<1x!quant.uniform<i8:f32, 3.0757404601899907E-5:-128>>
}


// -----

// CHECK-LABEL: @fold_reciprocal
func.func nested @fold_reciprocal() -> tensor<3x600x1200xf32> {
  // CHECK:           %[[VAL_0:.*]] = "tosa.const"() <{values = dense<8.620690e-03> : tensor<3x600x1200xf32>}> : () -> tensor<3x600x1200xf32>
  // CHECK:           return %[[VAL_0]] : tensor<3x600x1200xf32>
  // CHECK:         }
  %0 = "tosa.const"(){ values = dense<116.0>: tensor<3x600x1200xf32> }: () -> tensor<3x600x1200xf32>
  %1 = "tosa.reciprocal"(%0): (tensor<3x600x1200xf32>) -> tensor<3x600x1200xf32>
  return %1 : tensor<3x600x1200xf32>
}

// -----

// CHECK-LABEL: @do_not_fold_reciprocal_int
func.func nested @do_not_fold_reciprocal_int() -> tensor<3x600x1200xi32> {
  // CHECK:           tosa.reciprocal
  %0 = "tosa.const"(){ values = dense<11>: tensor<3x600x1200xi32> }: () -> tensor<3x600x1200xi32>
  %1 = "tosa.reciprocal"(%0): (tensor<3x600x1200xi32>) -> tensor<3x600x1200xi32>
  return %1 : tensor<3x600x1200xi32>
}

// -----

// CHECK-LABEL: @do_not_fold_intdiv_division_by_0
func.func @do_not_fold_intdiv_division_by_0() -> tensor<1x24x2xi32> {
  // CHECK: tosa.intdiv
  %1 = "tosa.const"() <{values = dense<0> : tensor<1x24x2xi32>}> : () -> tensor<1x24x2xi32>
  %4 = "tosa.const"() <{values = dense<20> : tensor<1x24x2xi32>}> : () -> tensor<1x24x2xi32>
  %16 = tosa.intdiv %4, %1 : (tensor<1x24x2xi32>, tensor<1x24x2xi32>) -> tensor<1x24x2xi32>
  return %16 : tensor<1x24x2xi32>
}


// -----
// CHECK-LABEL:   func.func @slice_dynamic_size_static_output_canonicalize(
// CHECK-SAME:                     %[[ARG0:.*]]: tensor<2x60x59x?xf32>) -> tensor<2x60x58x?xf32> {
// CHECK:           %[[START:.*]] = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[SIZE:.*]] = tosa.const_shape  {values = dense<[2, 60, 58, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[SLICE:.*]] = tosa.slice %[[ARG0]], %[[START]], %[[SIZE]] : (tensor<2x60x59x?xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x60x58x?xf32>
// CHECK:           return %[[SLICE]]
func.func @slice_dynamic_size_static_output_canonicalize(%arg0: tensor<2x60x59x?xf32>) -> tensor<2x60x58x?xf32> {
    %0 = tosa.const_shape  {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.const_shape  {values = dense<[-1, 60, 58, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.slice %arg0, %0, %1 : (tensor<2x60x59x?xf32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<2x60x58x?xf32>
    return %2 : tensor<2x60x58x?xf32>
}

// -----

// CHECK-LABEL: @fold_mul_shift
// CHECK-DAG: "tosa.const"() <{values = dense<1> : tensor<i32>}> : () -> tensor<i32>
func.func @fold_mul_shift() -> tensor<i32> {
    %0 = "tosa.const"() <{values = dense<-23661> : tensor<i32>}> : () -> tensor<i32>
    %1 = "tosa.const"() <{values = dense<-33022> : tensor<i32>}> : () -> tensor<i32>
    %2 = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.mul %0, %1, %2 : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
    return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: @fold_mul_no_shift
// CHECK-DAG: "tosa.const"() <{values = dense<781333542> : tensor<i32>}> : () -> tensor<i32>
func.func @fold_mul_no_shift() -> tensor<i32> {
    %0 = "tosa.const"() <{values = dense<-23661> : tensor<i32>}> : () -> tensor<i32>
    %1 = "tosa.const"() <{values = dense<-33022> : tensor<i32>}> : () -> tensor<i32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.mul %0, %1, %2 : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
    return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: @no_fold_mul_result_exceeds_i32
// CHECK-DAG: %[[LHS:.*]] = "tosa.const"() <{values = dense<23661> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG: %[[RHS:.*]] = "tosa.const"() <{values = dense<330222> : tensor<i32>}> : () -> tensor<i32>
// CHECK-DAG: %[[SHIFT:.*]] = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: tosa.mul %[[LHS]], %[[RHS]], %[[SHIFT]] : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
func.func @no_fold_mul_result_exceeds_i32() -> tensor<i32> {
    %0 = "tosa.const"() <{values = dense<23661> : tensor<i32>}> : () -> tensor<i32>
    %1 = "tosa.const"() <{values = dense<330222> : tensor<i32>}> : () -> tensor<i32>
    %2 = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = tosa.mul %0, %1, %2 : (tensor<i32>, tensor<i32>, tensor<1xi8>) -> tensor<i32>
    return %3 : tensor<i32>
}

// -----

// CHECK-LABEL: @test_fold_i1_to_i32_cast
// CHECK: %[[OUT:.*]] = "tosa.const"() <{values = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK: return %[[OUT]] : tensor<i32>
func.func @test_fold_i1_to_i32_cast() -> tensor<i32> {
  %0 = "tosa.const"() <{values = dense<1> : tensor<i1>}> : () -> tensor<i1>
  %1 = "tosa.cast"(%0) : (tensor<i1>) -> tensor<i32>
  return %1 : tensor<i32>
}

// -----

// CHECK-LABEL: @test_fold_i32_to_i1_cast
// CHECK: %[[OUT:.*]] = "tosa.const"() <{values = dense<true> : tensor<i1>}> : () -> tensor<i1>
// CHECK: return %[[OUT]] : tensor<i1>
func.func @test_fold_i32_to_i1_cast() -> tensor<i1> {
  %0 = "tosa.const"() <{values = dense<10> : tensor<i32>}> : () -> tensor<i32>
  %1 = "tosa.cast"(%0) : (tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

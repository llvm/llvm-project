// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s


// -----
// CHECK-LABEL: argmax
func @test_argmax(%arg0: tensor<14x19xf32>) -> tensor<14xi32> {
  %0 = "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<14x19xf32>) -> tensor<14xi32>
  return %0 : tensor<14xi32>
}

// -----
// CHECK-LABEL: avg_pool2d
func @test_avg_pool2d(%arg0: tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32> {
    %0 = "tosa.avg_pool2d"(%arg0) {kernel = [2, 2], pad = [0, 1, 0, 1], stride = [1, 1]} : (tensor<1x7x7x9xf32>) -> tensor<1x7x7x9xf32>
    return %0 : tensor<1x7x7x9xf32>
}

// -----
// CHECK-LABEL: conv2d
func @test_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<8x1x1x4xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<8x1x1x4xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
    return %0 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: depthwise_conv2d
func @test_depthwise_conv2d(%arg0: tensor<1x4x4x4xf32>, %arg1: tensor<1x1x4x2xf32>, %arg2: tensor<8xf32>) -> tensor<1x4x4x8xf32> {
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x4x4x4xf32>, tensor<1x1x4x2xf32>, tensor<8xf32>) -> tensor<1x4x4x8xf32>
  return %2 : tensor<1x4x4x8xf32>
}

// -----
// CHECK-LABEL: fully_connected
func @test_fully_connected(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>, %arg2: tensor<28xf32>) -> tensor<14x28xf32> {
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<14x19xf32>, tensor<19x28xf32>, tensor<28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

// -----
// CHECK-LABEL: test_matmul
func @test_matmul(%arg0: tensor<14x19xf32>, %arg1: tensor<19x28xf32>) -> tensor<14x28xf32> {
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<14x19xf32>, tensor<19x28xf32>) -> tensor<14x28xf32>
  return %0 : tensor<14x28xf32>
}

// -----
// CHECK-LABEL: max_pool2d
func @test_max_pool2d(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32> {
  %0 = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x32x32x8xf32>) -> tensor<1x32x32x8xf32>
  return %0 : tensor<1x32x32x8xf32>
}

// -----
/// CHECK-LABEL: transpose_conv2d
func @test_transpose_conv2d(%arg0: tensor<1x32x32x8xf32>, %arg1: tensor<16x1x1x8xf32>, %arg2: tensor<16xf32>) -> tensor<1x32x32x16xf32> {
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], out_shape = [1, 32, 32, 16], stride = [1, 1]} : (tensor<1x32x32x8xf32>, tensor<16x1x1x8xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  return %0 : tensor<1x32x32x16xf32>
}

// -----
// CHECK-LABEL: clamp
func @test_clamp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.clamp"(%arg0) {min_fp = 0.0 : f32, max_fp = 1.0: f32, min_int = 0 : i64, max_int = 1 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: relu
func @test_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}


// -----
// CHECK-LABEL: sigmoid
func @test_sigmoid(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.sigmoid"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: tanh
func @test_tanh(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.tanh"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: add
func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: arithmetic_right_shift
func @test_arithmetic_right_shift(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.arithmetic_right_shift"(%arg0, %arg1) { round = false } : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}


// -----
// CHECK-LABEL: bitwise_and
func @test_bitwise_and(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.bitwise_and"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: bitwise_or
func @test_bitwise_or(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x1x3xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.bitwise_or"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x1x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: bitwise_xor
func @test_bitwise_xor(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.bitwise_xor"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_and
func @test_logical_and(%arg0: tensor<13x21x3xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x3xi1> {
  %0 = "tosa.logical_and"(%arg0, %arg1) : (tensor<13x21x3xi1>, tensor<13x21x1xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: logical_left_shift
func @test_logical_left_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.logical_left_shift"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_right_shift
func @test_logical_right_shift(%arg0: tensor<13x21x3xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.logical_right_shift"(%arg0, %arg1) : (tensor<13x21x3xi32>, tensor<13x21x1xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: logical_or
func @test_logical_or(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = "tosa.logical_or"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: logical_xor
func @test_logical_xor(%arg0: tensor<13x1x3xi1>, %arg1: tensor<13x21x3xi1>) -> tensor<13x21x3xi1> {
  %0 = "tosa.logical_xor"(%arg0, %arg1) : (tensor<13x1x3xi1>, tensor<13x21x3xi1>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: maximum
func @test_max(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.maximum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: minimum
func @test_min(%arg0: tensor<13x21x3xf32>, %arg1: tensor<1x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.minimum"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<1x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: mul
func @test_mul(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.mul"(%arg0, %arg1)  { shift = 1 : i32 } : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: pow
func @test_pow(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.pow"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x1xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: sub
func @test_sub(%arg0: tensor<1x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.sub"(%arg0, %arg1) : (tensor<1x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: table
func @main(%arg0: tensor<64xi32>, %arg1: tensor<513x!quant.uniform<i16:f32, 1.0:0>>) -> tensor<64x!quant.uniform<i16:f32, 1.0:0>> {
    %0 = "tosa.table"(%arg0, %arg1) : (tensor<64xi32>, tensor<513x!quant.uniform<i16:f32, 1.0:0>>) -> tensor<64x!quant.uniform<i16:f32, 1.0:0>>
    return %0 : tensor<64x!quant.uniform<i16:f32, 1.0:0>>
}

// -----
// CHECK-LABEL: abs
func @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.abs"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: bitwise_not
func @test_bitwise_not(%arg0: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %0 = "tosa.bitwise_not"(%arg0) : (tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  return %0 : tensor<13x21x1xi32>
}

// -----
// CHECK-LABEL: ceil
func @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.ceil"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: clz
func @test_clz(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.clz"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: exp
func @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: floor
func @test_floor(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.floor"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: log
func @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.log"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: logical_not
func @test_logical_not(%arg0: tensor<1x21x3xi1>) -> tensor<1x21x3xi1> {
  %0 = "tosa.logical_not"(%arg0) : (tensor<1x21x3xi1>) -> tensor<1x21x3xi1>
  return %0 : tensor<1x21x3xi1>
}

// -----
// CHECK-LABEL: negate
func @test_negate(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.negate"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: reciprocal
func @test_reciprocal(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.reciprocal"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: rsqrt
func @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.rsqrt"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: select
func @test_select(%arg0: tensor<1x1x1xi1>, %arg1: tensor<13x21x3xf32>, %arg2: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x1x1xi1>, tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}


// -----
// CHECK-LABEL: equal
func @test_equal(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x1x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tosa.equal"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x1x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: greater
func @test_greater(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tosa.greater"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: greater_equal
func @test_greater_equal(%arg0: tensor<13x1x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xi1> {
  %0 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<13x1x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xi1>
  return %0 : tensor<13x21x3xi1>
}

// -----
// CHECK-LABEL: reduce_all
func @test_reduce_all(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xi1>) -> tensor<21x3xi1>
  return %1 : tensor<21x3xi1>
}

// -----
// CHECK-LABEL: reduce_any
func @test_reduce_any(%arg0: tensor<13x21x3xi1>) -> tensor<21x3xi1> {
  %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xi1>) -> tensor<1x21x3xi1>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xi1>) -> tensor<21x3xi1>
  return %1 : tensor<21x3xi1>
}

// -----
// CHECK-LABEL: reduce_max
func @test_reduce_max(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_min
func @test_reduce_min(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_product
func @test_reduce_product(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: reduce_sum
func @test_reduce_sum(%arg0: tensor<13x21x3xf32>) -> tensor<21x3xf32> {
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<1x21x3xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [21, 3]} : (tensor<1x21x3xf32>) -> tensor<21x3xf32>
  return %1 : tensor<21x3xf32>
}

// -----
// CHECK-LABEL: concat
func @test_concat(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<26x21x3xf32> {
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<26x21x3xf32>
  return %0 : tensor<26x21x3xf32>
}

// -----
// CHECK-LABEL: pad
func @test_pad(%arg0: tensor<13x21x3xf32>, %arg1: tensor<3x2xi32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.pad"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<3x2xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: reshape
func @test_reshape(%arg0: tensor<13x21x3xf32>) -> tensor<1x819xf32> {
  %0 = "tosa.reshape"(%arg0) {new_shape = [1, 819]} : (tensor<13x21x3xf32>) -> tensor<1x819xf32>
  return %0 : tensor<1x819xf32>
}

// -----
// CHECK-LABEL: reverse
func @test_reverse(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: slice
func @test_slice(%arg0: tensor<13x21x3xf32>) -> tensor<4x11x1xf32> {
  %0 = "tosa.slice"(%arg0) {start = [6, 8, 0], size = [4, 11, 1]} : (tensor<13x21x3xf32>) -> tensor<4x11x1xf32>
  return %0 : tensor<4x11x1xf32>
}

// -----
// CHECK-LABEL: tile
func @test_tile(%arg0: tensor<13x21x3xf32>) -> tensor<39x21x6xf32> {
  %0 = "tosa.tile"(%arg0) {multiples = [3, 1, 2]} : (tensor<13x21x3xf32>) -> tensor<39x21x6xf32>
  return %0 : tensor<39x21x6xf32>
}

// -----
// CHECK-LABEL: transpose
func @test_transpose(%arg0: tensor<13x21x3xf32>) -> tensor<3x13x21xf32> {
  %0 = "tosa.const"() {value = dense<[2, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<13x21x3xf32>, tensor<3xi32>) -> tensor<3x13x21xf32>
  return %1 : tensor<3x13x21xf32>
}

// -----
// CHECK-LABEL: gather
func @test_gather(%arg0: tensor<13x21x3xi32>, %arg1: tensor<26xi32>) -> tensor<26x21x3xi32> {
  %0 = "tosa.gather"(%arg0, %arg1) {axis = 0 : i32, batch_dims = 0 : i64} : (tensor<13x21x3xi32>, tensor<26xi32>) -> tensor<26x21x3xi32>
  return %0 : tensor<26x21x3xi32>
}

// Test TBD
// DISABLED-CHECK-LABEL: resize
//func @test_resize(%arg0: tensor<1x32x32x8xf32>) -> tensor<1x64x64x8xf32> {
//  %0 = "tosa.const"() {value = dense<64> : tensor<2xi32>} : () -> tensor<2xi32>
//  %1 = "tosa.resize"(%arg0, %0) {align_corners = false, half_pixel_centers = true} : (tensor<1x32x32x8xf32>, tensor<2xi32>) -> tensor<1x64x64x8xf32>
//  return %1 : tensor<1x64x64x8xf32>
//}

// -----
// CHECK-LABEL: cast
func @test_cast1(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xf32> {
  %0 = "tosa.cast"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xf32>
  return %0 : tensor<13x21x3xf32>
}

// -----
// CHECK-LABEL: cast2
func @test_cast2(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>> {
  %0 = "tosa.cast"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>>
  return %0 : tensor<13x21x3x!quant.uniform<u8:f32, 0.078431375324726104:128>>
}

// -----
// CHECK-LABEL: cast3
func @test_cast3(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>> {
  %0 = "tosa.cast"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
  return %0 : tensor<13x21x3x!quant.uniform<i16:f32, 0.078431375324726104:128>>
}

// -----
// CHECK-LABEL: rescale
func @test_rescale(%arg0: tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>> {
    %0 = "tosa.rescale"(%arg0) {double_round = false, input_zp = 127 : i32, multiplier = [1073741824 : i32], output_zp = -1 : i32, per_channel = false, scale32 = true, shift = [30 : i32]} : (tensor<13x21x3x!quant.uniform<u8:f32, 0.015655439347028732:127>>) -> tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
    return %0 : tensor<13x21x3x!quant.uniform<i8:f32, 0.015655439347028732:-1>>
}

// -----
// CHECK-LABEL: const
func @test_const(%arg0 : index) -> tensor<4xi32> {
    %0 = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: identity
func @test_identity(%arg0: tensor<13x21x3xi32>) -> tensor<13x21x3xi32> {
  %0 = "tosa.identity"(%arg0) : (tensor<13x21x3xi32>) -> tensor<13x21x3xi32>
  return %0 : tensor<13x21x3xi32>
}

// -----
// CHECK-LABEL: identityn
func @test_identityn(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0:2 = "tosa.identityn"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> (tensor<1xi32>, tensor<1xi32>)
  return %0#0 : tensor<1xi32>
}

// -----
// CHECK-LABEL: placeholder
func @test_placeholder() -> tensor<1xi32> {
  %0 = "tosa.placeholder"() : () -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// -----
// CHECK-LABEL: cond_if
func @test_cond_if(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i1>) -> tensor<f32> {
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ( {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %1 = "tosa.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tosa.yield"(%1) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %1 = "tosa.sub"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tosa.yield"(%1) : (tensor<f32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----
// CHECK-LABEL: while_loop
func @test_while_loop(%arg0: tensor<10xi32>, %arg1: tensor<i32>) {
  %0 = "tosa.const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1:3 = "tosa.while_loop"(%0, %0, %arg0) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<10xi32>):  // no predecessors
    %2 = "tosa.greater_equal"(%arg3, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.logical_not"(%2) : (tensor<i1>) -> tensor<i1>
    "tosa.yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<10xi32>):  // no predecessors
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %3 = "tosa.add"(%arg3, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %4 = "tosa.reshape"(%2) {new_shape = [1]} : (tensor<i32>) -> tensor<1xi32>
    %5 = "tosa.add"(%arg4, %4) : (tensor<10xi32>, tensor<1xi32>) -> tensor<10xi32>
    %6 = "tosa.add"(%arg2, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tosa.yield"(%6, %3, %5) : (tensor<i32>, tensor<i32>, tensor<10xi32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<10xi32>) -> (tensor<i32>, tensor<i32>, tensor<10xi32>)
  return
}

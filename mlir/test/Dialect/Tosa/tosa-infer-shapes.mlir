// RUN: mlir-opt --split-input-file --tosa-infer-shapes --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @test_return
func.func @test_return(%arg0 : tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: [[LOG:%.+]] = tosa.log %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tensor.cast [[LOG]] : tensor<4xf32> to tensor<*xf32>
  %0 = tosa.log %arg0 : (tensor<4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_multiple
func.func @test_multiple(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>, %arg2 : tensor<1xf32>) -> tensor<*xf32> {
  // CHECK: [[ADD:%.+]] = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: [[LOG:%.+]] = tosa.log %0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.log %0 : (tensor<*xf32>) -> tensor<*xf32>

  // CHECK: [[SUB:%.+]] = tosa.sub %0, %arg2 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %2 = tosa.sub %0, %arg2 : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_unary_f32
func.func @test_unary_f32(%arg0 : tensor<4xf32>) -> () {
  // CHECK: tosa.abs %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %0 = tosa.abs %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.ceil %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %1 = tosa.ceil %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.clamp %arg0 {{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %2 = tosa.clamp %arg0 { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.exp %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %3 = tosa.exp %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.floor %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %4 = tosa.floor %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.log %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %5 = tosa.log %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.negate %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %6 = tosa.negate %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.reciprocal %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %7 = tosa.reciprocal %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4xf32>) -> tensor<4xf32>
  %8 = tosa.reverse %arg0 { axis = 0 : i32 } : (tensor<4xf32>) -> tensor<?xf32>

  // CHECK: tosa.rsqrt %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %9 = tosa.rsqrt %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.tanh %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %10 = tosa.tanh %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.sigmoid %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %11 = tosa.sigmoid %arg0 : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: tosa.cast %arg0 : (tensor<4xf32>) -> tensor<4xi32>
  %12 = tosa.cast %arg0 : (tensor<4xf32>) -> tensor<*xi32>

  // CHECK: tosa.erf %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  %13 = tosa.erf %arg0 : (tensor<4xf32>) -> tensor<*xf32>
  return
}

// -----

// CHECK-LABEL: @test_unary_i32
func.func @test_unary_i32(%arg0 : tensor<4xi32>) -> () {
  // CHECK: tosa.abs %arg0 : (tensor<4xi32>) -> tensor<4xi32>
  %0 = tosa.abs %arg0 : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: tosa.bitwise_not %arg0 : (tensor<4xi32>) -> tensor<4xi32>
  %1 = tosa.bitwise_not %arg0 : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: tosa.clamp %arg0 {{.+}} : (tensor<4xi32>) -> tensor<4xi32>
  %2 = tosa.clamp %arg0 { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: tosa.clz %arg0 : (tensor<4xi32>) -> tensor<4xi32>
  %3 = tosa.clz %arg0 : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: tosa.negate %arg0 : (tensor<4xi32>) -> tensor<4xi32>
  %4 = tosa.negate %arg0 : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<4xi32>
  %5 = tosa.reverse %arg0 { axis = 0 : i32 } : (tensor<4xi32>) -> tensor<?xi32>

  // CHECK: tosa.rescale %arg0 {{.+}} : (tensor<4xi32>) -> tensor<4xi16>
  %6 = tosa.rescale %arg0 {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = array<i32: 42, 43>, shift = array<i8: 14, 15>, scale32 = false, double_round = false, per_channel = false} : (tensor<4xi32>) -> tensor<*xi16>

  // CHECK: tosa.identity %arg0 : (tensor<4xi32>) -> tensor<4xi32>
  %7 = tosa.identity %arg0 : (tensor<4xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_binary_scalar_f32
func.func @test_binary_scalar_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>) -> () {
  // CHECK: tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %1 = tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %2 = tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.mul %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %3 = tosa.mul %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %4 = tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %5 = tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %6 = tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %7 = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %8 = tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_binary_broadcast_f32
func.func @test_binary_broadcast_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>) -> () {
  // CHECK: tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %0 = tosa.add %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %1 = tosa.maximum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %2 = tosa.minimum %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.mul %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %3 = tosa.mul %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %4 = tosa.pow %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %5 = tosa.sub %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %6 = tosa.equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %7 = tosa.greater %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %8 = tosa.greater_equal %arg0, %arg1 : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_binary_i32
func.func @test_binary_i32(%arg0 : tensor<4xi32>, %arg1 : tensor<1xi32>) -> () {
  // CHECK: tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %0 = tosa.add %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.bitwise_and %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %1 = tosa.bitwise_and %arg0, %arg1: (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.bitwise_or %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %2 = tosa.bitwise_or %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.bitwise_xor %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %3 = tosa.bitwise_xor %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi1>
  %4 = tosa.equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>

  // CHECK: tosa.greater %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi1>
  %5 = tosa.greater %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>

  // CHECK: tosa.greater_equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi1>
  %6 = tosa.greater_equal %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi1>

  // CHECK: tosa.logical_left_shift %arg0, %arg1 {shift = 0 : i32} : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %7 = tosa.logical_left_shift %arg0, %arg1 { shift = 0 : i32 }: (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.logical_right_shift %arg0, %arg1 {shift = 0 : i32} : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %8 = tosa.logical_right_shift %arg0, %arg1 { shift = 0 : i32 }: (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.maximum %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %9 = tosa.maximum %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.minimum %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %10 = tosa.minimum %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK: tosa.mul %arg0, %arg1, %{{.*}} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<4xi32>
  %shift = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %11 = tosa.mul %arg0, %arg1, %shift : (tensor<4xi32>, tensor<1xi32>, tensor<1xi8>) -> tensor<*xi32>

  // CHECK: tosa.pow %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %12 = tosa.pow %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  // CHECK:  tosa.sub %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<4xi32>
  %13 = tosa.sub %arg0, %arg1 : (tensor<4xi32>, tensor<1xi32>) -> tensor<*xi32>

  return
}

// -----

// CHECK-LABEL: @test_binary_i1
func.func @test_binary_i1(%arg0 : tensor<4xi1>, %arg1 : tensor<1xi1>) -> () {
  // CHECK: tosa.logical_and %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<4xi1>
  %0 = tosa.logical_and %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<*xi1>

  // CHECK: tosa.logical_or %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<4xi1>
  %1 = tosa.logical_or %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<*xi1>

  // CHECK: tosa.logical_xor %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<4xi1>
  %2 = tosa.logical_xor %arg0, %arg1 : (tensor<4xi1>, tensor<1xi1>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_select_i32
func.func @test_select_i32(%arg0 : tensor<4xi1>, %arg1 : tensor<1xi32>, %arg2 : tensor<4xi32>) -> () {
  // CHECK: tosa.select %arg0, %arg1, %arg2 : (tensor<4xi1>, tensor<1xi32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = tosa.select %arg0, %arg1, %arg2 : (tensor<4xi1>, tensor<1xi32>, tensor<4xi32>) -> tensor<*xi32>

  return
}

// -----

// CHECK-LABEL: @test_static_argmax
func.func @test_static_argmax(%arg0 : tensor<2x3xi32>) -> () {
  // CHECK: tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<3xi32>
  %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<?xi32>

  // CHECK: tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<2xi32>
  %1 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x3xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_dynamic_argmax
func.func @test_dynamic_argmax(%arg0 : tensor<2x?xi32>) -> () {
  // CHECK: tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x?xi32>) -> tensor<?xi32>
  %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<2x?xi32>) -> tensor<?xi32>

  // CHECK: tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x?xi32>) -> tensor<2xi32>
  %1 = tosa.argmax %arg0 {axis = 1 : i32} : (tensor<2x?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_static_fully_connected
func.func @test_static_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_input_fully_connected
func.func @test_static_input_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<3x?xf32>
  %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_weight_fully_connected
func.func @test_static_weight_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x5xf32>
  %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_bias_fully_connected
func.func @test_static_bias_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x5xf32>
  %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_out_fully_connected
func.func @test_static_out_fully_connected(%arg0 : tensor<3x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = tosa.fully_connected %arg0, %arg1, %arg2 : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_matmul
func.func @test_static_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_lhs_matmul
func.func @test_dynamic_lhs_matmul(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: tosa.matmul %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<2x?x5xi32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_rhs_matmul
func.func @test_dynamic_rhs_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<?x?x?xi32>) -> () {
  // CHECK: tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<2x3x?xi32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_mixed_matmul
func.func @test_dynamic_mixed_matmul(%arg0 : tensor<?x3x?xi32>, %arg1 : tensor<?x?x5xi32>) -> () {
  // CHECK: tosa.matmul %arg0, %arg1 : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x3x5xi32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_table_static
func.func @test_table_static(%arg0 : tensor<4x5xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:tosa.table %arg0, %arg1 : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<4x5xi16>
  %0 = tosa.table %arg0, %arg1 : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABEL: @test_table_dynamic
func.func @test_table_dynamic(%arg0 : tensor<4x?xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:tosa.table %arg0, %arg1 : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<4x?xi16>
  %0 = tosa.table %arg0, %arg1 : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABEL: @test_static_reshape
func.func @test_static_reshape(%arg0 : tensor<4x4xi32>) -> () {
  // CHECK: tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x4xi32>) -> tensor<16xi32>
  %0 = tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x4xi32>) -> tensor<?xi32>

  // CHECK: tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x4xi32>) -> tensor<16xi32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x4xi32>) -> tensor<?xi32>

  // CHECK: tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x4xi32>) -> tensor<2x8xi32>
  %2 = tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x4xi32>) -> tensor<?x?xi32>

  return
}
// -----

// CHECK-LABEL: @test_dynamic_reshape
func.func @test_dynamic_reshape(%arg0 : tensor<4x?xi32>) -> () {
  // CHECK: %0 = tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x?xi32>) -> tensor<16xi32>
  %0 = tosa.reshape %arg0 {new_shape = array<i64: 16>} : (tensor<4x?xi32>) -> tensor<?xi32>

  // CHECK: %1 = tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x?xi32>) -> tensor<?xi32>
  %1 = tosa.reshape %arg0 {new_shape = array<i64: -1>} : (tensor<4x?xi32>) -> tensor<?xi32>

  // CHECK: %2 = tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x?xi32>) -> tensor<2x?xi32>
  %2 = tosa.reshape %arg0 {new_shape = array<i64: 2, -1>} : (tensor<4x?xi32>) -> tensor<?x?xi32>

  return
}

// -----

// CHECK: @test_reduce_binary
func.func @test_reduce_binary(%arg0 : tensor<2x3x?x?xi1>) -> () {
  // CHECK: tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<1x3x?x?xi1>
  %0 = tosa.reduce_all %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xi1>) -> tensor<2x1x?x?xi1>
  %1 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xi1>) -> tensor<2x3x1x?xi1>
  %2 = tosa.reduce_all %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: tosa.reduce_all %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xi1>) -> tensor<2x3x?x1xi1>
  %3 = tosa.reduce_all %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<1x3x?x?xi1>
  %4 = tosa.reduce_any %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  return
}

// -----

// CHECK: @test_reduce_float
func.func @test_reduce_float(%arg0 : tensor<2x3x?x?xf32>) -> () {
  // CHECK: tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xf32>) -> tensor<1x3x?x?xf32>
  %0 = tosa.reduce_sum %arg0 {axis = 0 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x1x?x?xf32>
  %1 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x3x1x?xf32>
  %2 = tosa.reduce_sum %arg0 {axis = 2 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %3 = tosa.reduce_sum %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %4 = tosa.reduce_max %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %5 = tosa.reduce_min %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: tosa.reduce_prod %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %6 = tosa.reduce_prod %arg0 {axis = 3 : i32} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat
func.func @test_concat(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic
func.func @test_concat_dynamic(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x?xf32>) -> () {
  // CHECK: tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<3x2xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic_axis
func.func @test_concat_dynamic_axis(%arg0 : tensor<?x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_axis_1
func.func @test_concat_axis_1(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}


// -----

// CHECK-LABEL:@test_padding_dynamic_input
func.func @test_padding_dynamic_input(%arg0 : tensor<1x?xf32>) -> () {
  %0 = tosa.const_shape { value = dense<[1, 2, 3, 4]> : tensor<4xindex> } : () -> !tosa.shape<4>
  // CHECK: tosa.pad %arg0, %0 : (tensor<1x?xf32>, !tosa.shape<4>) -> tensor<4x?xf32>
  %1 = tosa.pad %arg0, %0  : (tensor<1x?xf32>, !tosa.shape<4>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_padding_simple
func.func @test_padding_simple(%arg0 : tensor<1x2xf32>) -> () {
  %0 = tosa.const_shape { value = dense<[1, 2, 3, 4]> : tensor<4xindex> } : () -> !tosa.shape<4>
  // CHECK: tosa.pad %arg0, %0 : (tensor<1x2xf32>, !tosa.shape<4>) -> tensor<4x9xf32>
  %1 = tosa.pad %arg0, %0  : (tensor<1x2xf32>, !tosa.shape<4>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_slice
func.func @test_slice(%arg0 : tensor<?xi32>) -> () {
  // CHECK: %0 = tosa.const_shape  {value = dense<1> : tensor<1xindex>}
  // CHECK: %1 = tosa.const_shape  {value = dense<2> : tensor<1xindex>}
  // CHECK: %2 = tosa.slice %arg0, %0, %1 : (tensor<?xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<2xi32>
  %0 = tosa.const_shape {value = dense<1> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = tosa.const_shape {value = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2= tosa.slice %arg0, %0, %1 : (tensor<?xi32>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_slice_size_minus_one
func.func @test_slice_size_minus_one(%arg0 : tensor<?x8x8x8xi32>) -> () {
  // CHECK: %[[Start:.+]] = tosa.const_shape
  // CHECK: %[[Size:.+]] = tosa.const_shape
  // CHECK: %[[VAL:.+]] = tosa.slice %arg0, %[[Start]], %[[Size]] : (tensor<?x8x8x8xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x7x?x?xi32>
  // this checks following
  //  dim 0: size=-1, input dim=? => inferred output dim is ?
  //  dim 1: size=-1 => inferred output dim is input_dim - start
  //  dim 2: size=-1, start=-1 => inferred output dim is ?
  //  dim 3: size=-1, start=8 => inferred output dim is ? because start is out of bound
  %start = tosa.const_shape {value = dense<[0, 1, -1, 8]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape {value = dense<[-1, -1, -1, -1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %2= tosa.slice %arg0, %start, %size : (tensor<?x8x8x8xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_slice_size_out_of_bound
func.func @test_slice_size_out_of_bound(%arg0 : tensor<8x8x8x?xi32>) -> () {
  // CHECK: %[[Start:.+]] = tosa.const_shape
  // CHECK: %[[Size:.+]] = tosa.const_shape
  // CHECK: %[[VAL:.+]] = tosa.slice %arg0, %[[Start]], %[[Size]] : (tensor<8x8x8x?xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x?x?x4xi32>
  // this checks following
  //  dim 0: size=0 => inferred output dim is ?
  //  dim 1: size=-2 => inferred output dim is ?
  //  dim 3: start+size out of bound because size too big: inferred output dim is ?
  //  dim 4: size=4, input dim=? => inferred output dim is 4
  %start = tosa.const_shape {value = dense<[0, 0, 0, 0]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape {value = dense<[0, -2, 9, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %2= tosa.slice %arg0, %start, %size : (tensor<8x8x8x?xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_slice_start_out_of_bound
func.func @test_slice_start_out_of_bound(%arg0 : tensor<8x8x8x?xi32>) -> () {
  // CHECK: %[[Start:.+]] = tosa.const_shape
  // CHECK: %[[Size:.+]] = tosa.const_shape
  // CHECK: %[[VAL:.+]] = tosa.slice %arg0, %[[Start]], %[[Size]] : (tensor<8x8x8x?xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x?x?x4xi32>
  // this checks following
  //  dim 0: start=-1 => inferred output dim is ?
  //  dim 1: start=8 => inferred output dim is ?
  //  dim 2: start+size out of bound: inferred output dim is ?
  //  dim 3: start=8000000, size=4, input dim=? => inferred output dim is 4
  %start = tosa.const_shape {value = dense<[-1, 8, 6, 8000000]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %size = tosa.const_shape {value = dense<[1, 1, 3, 4]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %2= tosa.slice %arg0, %start, %size : (tensor<8x8x8x?xi32>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<?x?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_slice_dynamic
func.func @test_slice_dynamic(%arg0 : tensor<10x?x2xf32>) -> () {
  // CHECK: %0 = tosa.const_shape  {value = dense<[1, 0, 0]> : tensor<3xindex>}
  // CHECK: %1 = tosa.const_shape  {value = dense<[7, -1, 1]> : tensor<3xindex>}
  // CHECK: %2 = tosa.slice %arg0, %0, %1 : (tensor<10x?x2xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<7x?x1xf32>
  %0 = tosa.const_shape {value = dense<[1, 0, 0]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %1 = tosa.const_shape {value = dense<[7, -1, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %2= tosa.slice %arg0, %0, %1 : (tensor<10x?x2xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_tile
func.func @test_tile(%arg0 : tensor<2x3x?xi32>) -> () {
  // CHECK: %[[CST:.*]] = tosa.const_shape {value = dense<[2, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  // CHECK: tosa.tile %arg0, %[[CST]] : (tensor<2x3x?xi32>, !tosa.shape<3>) -> tensor<4x3x?xi32>
  %cst = tosa.const_shape {value = dense<[2, 1, 5]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %0 = tosa.tile %arg0, %cst : (tensor<2x3x?xi32>, !tosa.shape<3>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_transpose_same
func.func @test_transpose_same(%arg0 : tensor<4x4x4xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: tosa.transpose %arg0, %arg1 : (tensor<4x4x4xi32>, tensor<3xi32>) -> tensor<4x4x4xi32>
  %0 = tosa.transpose %arg0, %arg1 : (tensor<4x4x4xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_transpose_perm_unknown
func.func @test_transpose_perm_unknown(%arg0 : tensor<4x4x5xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: tosa.transpose %arg0, %arg1 : (tensor<4x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %0 = tosa.transpose %arg0, %arg1 : (tensor<4x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_transpose_static
func.func @test_transpose_static(%arg0 : tensor<3x4x5xi32>) -> () {
  %0 = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
  // CHECK: tosa.transpose %arg0, %cst : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<5x4x3xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @gather_static
func.func @gather_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<3x6xi32>) -> tensor<3x6x5xi32>
  %0 = tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<3x6xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_values
func.func @gather_dynamic_values(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: tosa.gather %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<3x6xi32>) -> tensor<3x6x?xi32>
  %0 = tosa.gather %arg0, %arg1 : (tensor<?x?x?xi32>, tensor<3x6xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_indices
func.func @gather_dynamic_indices(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>) {
  // CHECK: tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<?x?xi32>) -> tensor<3x?x5xi32>
  %0 = tosa.gather %arg0, %arg1 : (tensor<3x4x5xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @gather_minimum_info
func.func @gather_minimum_info(%arg0 : tensor<3x?x5xi32>, %arg1 : tensor<?x6xi32>) {
  // CHECK: tosa.gather %arg0, %arg1 : (tensor<3x?x5xi32>, tensor<?x6xi32>) -> tensor<3x6x5xi32>
  %0 = tosa.gather %arg0, %arg1 : (tensor<3x?x5xi32>, tensor<?x6xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @scatter_static
func.func @scatter_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> tensor<3x4x5xi32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @scatter_static_values
func.func @scatter_static_values(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>) -> tensor<3x4x5xi32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @scatter_static_indices
func.func @scatter_static_indices(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>) -> tensor<3x?x?xi32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @scatter_static_input
func.func @scatter_static_input(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>) -> tensor<3x?x5xi32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @scatter_minimum_static
func.func @scatter_minimum_static(%arg0 : tensor<?x4x?xi32>, %arg1 : tensor<3x?xi32>, %arg2 : tensor<?x?x5xi32>) {
  // CHECK: tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>) -> tensor<3x4x5xi32>
  %0 = tosa.scatter %arg0, %arg1, %arg2 : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @test_pool_static
func.func @test_pool_static(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x2x4x7xf32>
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x2x4x7xf32>
  %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_static
func.func @conv2d_static(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_input
func.func @conv2d_dynamic_input(%input: tensor<?x?x?x?xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<?x?x?x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_dynamic_input
func.func @test_pool_dynamic_input(%arg0: tensor<?x?x?x?xf32>) {
  // CHECK: -> tensor<?x?x?x?xf32>
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<?x?x?x?xf32>
  %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_padded
func.func @test_pool_padded(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x5x11x7xf32>
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x5x11x7xf32>
  %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_weight
func.func @conv2d_dynamic_weight(%input: tensor<2x8x9x3xf32>, %weights: tensor<?x?x?x?xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x?x?x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_bias
func.func @conv2d_dynamic_bias(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<?xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_stride
func.func @test_pool_stride(%arg0: tensor<3x11x12x7xf32>) {
  // CHECK: -> tensor<3x4x4x7xf32>
  %0 = tosa.avg_pool2d %arg0 {acc_type = f32, kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x4x4x7xf32>
  %1 = tosa.max_pool2d %arg0 {kernel = array<i64: 4, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 3>} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_padded
func.func @conv2d_padded(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x9x11x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_dilated
func.func @conv2d_dilated(%input: tensor<2x12x14x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 3, 2>} : (tensor<2x12x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_strided
func.func @conv2d_strided(%input: tensor<1x13x14x1xf32>, %weights: tensor<1x1x1x1xf32>, %bias: tensor<1xf32>) -> () {
  // CHECK: -> tensor<1x5x7x1xf32>
  %0 = tosa.conv2d %input, %weights, %bias {acc_type = f32, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 2>, dilation = array<i64: 1, 1>} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_static
func.func @conv3d_static(%input: tensor<2x8x9x10x3xf32>, %weights: tensor<5x3x6x4x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x7x5xf32>
  %0 = tosa.conv3d %input, %weights, %bias {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_input
func.func @conv3d_dynamic_input(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<?x?x?x?x5xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<?x?x?x?x?xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_weight
func.func @conv3d_dynamic_weight(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x?x?x?x5xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<?x?x?x?x?xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_bias
func.func @conv3d_dynamic_bias(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x6x4x7x5xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_padded
func.func @conv3d_padded(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x9x11x18x5xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 1, 2, 3, 4, 5, 6>, stride = array<i64: 1, 1, 1>} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dilated
func.func @conv3d_dilated(%arg0: tensor<2x12x14x16x3xf32>, %arg1: tensor<5x3x6x2x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x6x4x12x5xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 3, 2, 4>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>} : (tensor<2x12x14x16x3xf32>, tensor<5x3x6x2x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_strided
func.func @conv3d_strided(%arg0: tensor<1x13x14x15x1xf32>, %arg1: tensor<1x1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x5x7x4x1xf32>
  %0 = tosa.conv3d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1, 1>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 3, 2, 4>} : (tensor<1x13x14x15x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_static
func.func @depthwise_conv2d_static(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_input
func.func @depthwise_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<?x?x?x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<?x?x?x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_weight
func.func @depthwise_conv2d_dynamic_weight(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x?x?x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<15xf32>) -> tensor<2x?x?x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_bias
func.func @depthwise_conv2d_dynamic_bias(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<?xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_padded
func.func @depthwise_conv2d_padded(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x9x11x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 2, 3, 4>, stride = array<i64: 1, 1>} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x9x11x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dilated
func.func @depthwise_conv2d_dilated(%arg0: tensor<2x12x14x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 3, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x12x14x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_strided
func.func @depthwise_conv2d_strided(%arg0: tensor<1x13x14x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x5x7x1xf32>
  %0 = tosa.depthwise_conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 3, 2>} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x5x7x1xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_out_shape
func.func @transpose_conv2d_out_shape(%arg0: tensor<2x?x?x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x8x9x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, 8, 9, -1>, stride = array<i64: 1, 1>} : (tensor<2x?x?x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x8x9x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_static
func.func @transpose_conv2d_static(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x18x19x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_static_strided
func.func @transpose_conv2d_static_strided(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x33x45x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 2, 3>} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_input
func.func @transpose_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<?x?x?x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_weights
func.func @transpose_conv2d_dynamic_weights(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x?x?x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x6x4x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_bias
func.func @transpose_conv2d_dynamic_bias(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x8x9x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x6x4x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>) -> tensor<2x8x9x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_padded
func.func @transpose_conv2d_padded(%arg0: tensor<2x9x11x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x10x13x5xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 1, 0, 3, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 1, 1>} : (tensor<2x9x11x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x10x13x5xf32>
  return
}

// CHECK-LABEL: @transpose_conv2d_strided
func.func @transpose_conv2d_strided(%arg0: tensor<1x5x7x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x13x13x1xf32>
  %0 = tosa.transpose_conv2d %arg0, %arg1, %arg2 {acc_type = f32, out_pad = array<i64: 0, 0, 0, 0>, out_shape = array<i64: -1, -1, -1, -1>, stride = array<i64: 3, 2>} : (tensor<1x5x7x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x13x13x1xf32>
  return
}

// -----

// CHECK-LABEL: @resize_int_horizontal
func.func @resize_int_horizontal(%arg0: tensor<1x15x13x1xi8>) {
  // CHECK: -> tensor<1x23x179x1xi8>
  %0 = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR", scale = array<i64: 11, 7, 89, 6>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x15x13x1xi8>) -> tensor<?x?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @resize_int_vertical
func.func @resize_int_vertical(%arg0: tensor<1x49x42x1xi16>) {
  // CHECK: -> tensor<1x112x220x1xi16>
  %0 = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR", scale = array<i64: 37, 16, 219, 41>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x49x42x1xi16>) -> tensor<?x?x?x?xi16>
  return
}

// -----

// CHECK-LABEL: @resize_int_power_of_two_upscale
func.func @resize_int_power_of_two_upscale(%arg0: tensor<1x23x19x1xi8>) {
  // CHECK: -> tensor<1x353x289x1xi32>
  %0 = tosa.resize %arg0 {mode = "BILINEAR", scale = array<i64: 16, 1, 16, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x23x19x1xi8>) -> tensor<?x?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @resize_int_power_of_two_upscale_offsetted
func.func @resize_int_power_of_two_upscale_offsetted(%arg0: tensor<1x41x26x1xi16>) {
  // CHECK: -> tensor<1x328x208x1xi48>
  %0 = tosa.resize %arg0 {mode = "BILINEAR", scale = array<i64: 16, 2, 16, 2>, offset = array<i64: -7, -7>, border = array<i64: 7, 7>} : (tensor<1x41x26x1xi16>) -> tensor<?x?x?x?xi48>
  return
}

// -----
// CHECK-LABEL: @resize_fp_horizontal
func.func @resize_fp_horizontal(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x106x85x1xf32>
  %0 = tosa.resize %arg0 {mode = "BILINEAR", scale = array<i64: 15, 7, 84, 47>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----
// CHECK-LABEL: @resize_fp_vertical
func.func @resize_fp_vertical(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x128x13x1xf32>
  %0 = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR", scale = array<i64: 127, 49, 12, 47>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @resize_fp_power_of_two_upscale
func.func @resize_fp_power_of_two_upscale(%arg0: tensor<1x23x23x1xf32>) {
  // CHECK: -> tensor<1x89x89x1xf32>
  %0 = tosa.resize %arg0 {mode = "BILINEAR", scale = array<i64: 4, 1, 4, 1>, offset = array<i64: 0, 0>, border = array<i64: 0, 0>} : (tensor<1x23x23x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @resize_fp_power_of_two_upscale_offsetted
func.func @resize_fp_power_of_two_upscale_offsetted(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x1600x1536x1xf32>
  %0 = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR", scale = array<i64: 64, 2, 64, 2>, offset = array<i64: -31, -31>, border = array<i64: 31, 31>} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @if_test_simple
func.func @if_test_simple(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> () {
  %a = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
  %b = tosa.log %arg1 : (tensor<f32>) -> tensor<f32>

  // CHECK: tosa.cond_if
  // CHECK: -> (tensor<f32>)
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    tosa.yield %a : tensor<f32>
  } else {
    tosa.yield %b : tensor<f32>
  }
  return
}

// -----

// CHECK-LABEL: @if_test_dynamic
func.func @if_test_dynamic(%arg0 : tensor<2xf32>, %arg1 : tensor<3xf32>, %arg2 : tensor<i1>) -> () {
  // CHECK: tosa.cond_if
  // CHECK: -> (tensor<?xf32>)
  %0 = tosa.cond_if %arg2 -> (tensor<?xf32>) {
    tosa.yield %arg0 : tensor<2xf32>
  } else {
    tosa.yield %arg1 : tensor<3xf32>
  }
  return
}

// -----

// CHECK-LABEL: @if_test_unranked
func.func @if_test_unranked(%arg0 : tensor<f32>, %arg1 : tensor<3xf32>, %arg2 : tensor<i1>) -> () {
  // CHECK: tosa.cond_if
  // CHECK: -> (tensor<*xf32>)
  %0 = tosa.cond_if %arg2 -> (tensor<*xf32>) {
    tosa.yield %arg0 : tensor<f32>
  } else {
    tosa.yield %arg1 : tensor<3xf32>
  }
  return
}

// -----

// CHECK-LABEL: @if_test_propagate
func.func @if_test_propagate(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> () {
  // CHECK: tosa.cond_if
  // CHECK: -> (tensor<f32>)
  %0 = tosa.cond_if %arg2 -> (tensor<f32>) {
    %1 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  } else {
    %1 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    tosa.yield %1 : tensor<f32>
  }
  return
}

// -----

// CHECK-LABEL: @while_test
func.func @while_test(%arg0 : tensor<i32>) -> (tensor<*xi32>) {
  // CHECK:      tosa.add
  // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = tosa.add %arg0, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<*xi32>

  // CHECK:      tosa.while_loop
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  %1 = tosa.while_loop (%arg1 = %0) : (tensor<*xi32>) -> tensor<*xi32> {
    %2 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>

    // CHECK:       tosa.greater_equal
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.greater_equal %2, %arg1 : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>

    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i1>
    tosa.yield %3 : tensor<*xi1>

  } do {

  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  ^bb0(%arg1: tensor<*xi32>):
    %2 = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>

    // CHECK:     tosa.add
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.add %arg1, %2 : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>

    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i32>
    tosa.yield %3 : tensor<*xi32>
  }

  // CHECK:      tensor.cast
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @while_test
func.func @while_test(%arg0 : tensor<i32>, %arg1 : tensor<1xi32>) -> () {
  // CHECK:      tosa.while_loop
  // CHECK-SAME: (tensor<i32>, tensor<1xi32>) -> (tensor<i32>, tensor<?xi32>)
  %0:2 = tosa.while_loop (%arg2 = %arg0, %arg3 = %arg1) : (tensor<i32>, tensor<1xi32>) -> (tensor<i32>, tensor<?xi32>) {
    %1 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    // CHECK:       tosa.greater_equal
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = tosa.greater_equal %1, %arg2 : (tensor<i32>, tensor<i32>) -> tensor<i1>

    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i1>
    tosa.yield %2 : tensor<i1>
  } do {

  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  // CHECK-SAME: tensor<?xi32>
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<?xi32>):
    %1 = "tosa.const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>

    // CHECK:     tosa.add
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = tosa.add %arg2, %1 : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // CHECK:      tosa.concat
    // CHECK-SAME: (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = tosa.concat %arg3, %arg3 {axis = 0 : i32} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>

    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i32>
    // CHECK-SAME: tensor<?xi32>
    tosa.yield %2, %3 : tensor<i32>, tensor<?xi32>
  }
  return
}

// -----

// This test locks down a fix for a crash in the type inference process.
// The relevant pattern is a while loop whose body contains a TOSA operation which is
// consumed by a non-inferrable user in the same body.
// Previously, this would trigger a crash due to how types are cached and then
// reapplied to the operations in the loops body.

// CHECK-LABEL: @while_dont_crash
func.func @while_dont_crash(%arg0 : tensor<i32>) -> (tensor<*xi32>) {
  %0 = tosa.add %arg0, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
  // CHECK:      tosa.while_loop
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  %1 = tosa.while_loop (%arg1 = %0) : (tensor<*xi32>) -> tensor<*xi32> {
    %2 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    // CHECK:       tosa.greater_equal
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.greater_equal %2, %arg1 : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
    tosa.yield %3 : tensor<*xi1>
  } do {
  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  ^bb0(%arg1: tensor<*xi32>):
    // CHECK:     tosa.add
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = tosa.add %arg1, %arg1 : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    // CHECK: %[[CAST:.+]] = tensor.cast %{{.*}} : tensor<i32> to tensor<*xi32>
    // CHECK: "use"(%[[CAST]]) : (tensor<*xi32>) -> ()
    "use"(%3) : (tensor<*xi32>) -> ()
    tosa.yield %3 : tensor<*xi32>
  }
  // CHECK: tensor.cast
  return %1 : tensor<*xi32>
}

// -----

// This test locks down a fix for a crash in the type inference process.
// The relevant pattern is a while loop whose body contains a TOSA operation which is
// consumed by a non-inferrable user in the same body.

// CHECK-LABEL: @while_dont_crash_nested
func.func @while_dont_crash_nested(%arg0 : tensor<i32>) -> (tensor<*xi32>) {
  %0 = tosa.add %arg0, %arg0 : (tensor<i32>, tensor<i32>) -> tensor<*xi32>
  // CHECK:      tosa.while_loop
  // CHECK-SAME: (tensor<i32>) -> tensor<i32>
  %1 = tosa.while_loop (%arg1 = %0) : (tensor<*xi32>) -> tensor<*xi32> {
    %2 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
    // CHECK:       tosa.greater_equal
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = tosa.greater_equal %2, %arg1 : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i1>
    tosa.yield %3 : tensor<*xi1>
  } do {
  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  ^bb0(%arg1: tensor<*xi32>):
    // CHECK:      tosa.while_loop
    // CHECK-SAME: (tensor<i32>) -> tensor<i32>
    %1 = tosa.while_loop (%arg2 = %arg1) : (tensor<*xi32>) -> tensor<*xi32> {
      %2 = "tosa.const"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
      // CHECK:       tosa.greater_equal
      // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
      %4 = tosa.greater_equal %2, %arg2 : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
      // CHECK:      tosa.yield
      // CHECK-SAME: tensor<i1>
      tosa.yield %4 : tensor<*xi1>
    } do {
    // CHECK:      ^bb0
    // CHECK-SAME: tensor<i32>
    ^bb0(%arg2: tensor<*xi32>):
      // CHECK:     tosa.add
      // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
      %4 = tosa.add %arg2, %arg2 : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      // CHECK: %[[CAST:.+]] = tensor.cast %{{.*}} : tensor<i32> to tensor<*xi32>
      // CHECK: "use"(%[[CAST]]) : (tensor<*xi32>) -> ()
      "use"(%4) : (tensor<*xi32>) -> ()
      // CHECK:      tosa.yield
      // CHECK-SAME: tensor<i32>
      tosa.yield %4 : tensor<*xi32>
    }
    // CHECK:      tosa.yield
    // CHECK-SAME: tensor<i32>
    tosa.yield %1 : tensor<*xi32>
  }

  // CHECK: tensor.cast
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @test_static_rfft2d
func.func @test_static_rfft2d(%arg0: tensor<5x2x8xf32>) -> () {
  // CHECK: -> (tensor<5x2x5xf32>, tensor<5x2x5xf32>)
  %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<5x2x8xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_dynamic_batch_rfft2d
func.func @test_dynamic_batch_rfft2d(%arg0 : tensor<?x2x4xf32>) -> () {
  // CHECK: -> (tensor<?x2x3xf32>, tensor<?x2x3xf32>)
  %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<?x2x4xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_dynamic_width_rfft2d
func.func @test_dynamic_width_rfft2d(%arg0 : tensor<5x2x?xf32>) -> () {
  // CHECK: -> (tensor<5x2x?xf32>, tensor<5x2x?xf32>)
  %output_real, %output_imag = tosa.rfft2d %arg0 : (tensor<5x2x?xf32>) -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_static_fft2d
func.func @test_static_fft2d(%arg0: tensor<1x4x8xf32>, %arg1: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  // CHECK: -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<1x4x8xf32>, tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>)
  return %output_real, %output_imag : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// -----

// CHECK-LABEL: @test_dynamic_batch_fft2d
func.func @test_dynamic_batch_fft2d(%arg0: tensor<?x4x8xf32>, %arg1: tensor<?x4x8xf32>) -> (tensor<?x4x8xf32>, tensor<?x4x8xf32>) {
  // CHECK: -> (tensor<?x4x8xf32>, tensor<?x4x8xf32>)
  %output_real, %output_imag = tosa.fft2d %arg0, %arg1 {inverse = false} : (tensor<?x4x8xf32>, tensor<?x4x8xf32>) -> (tensor<?x4x8xf32>, tensor<?x4x8xf32>)
  return %output_real, %output_imag : tensor<?x4x8xf32>, tensor<?x4x8xf32>
}

// -----

// CHECK-LABEL: @test_unranked_equal
func.func @test_unranked_equal(%arg0 : tensor<*xf32>, %arg1 : tensor<f32>) -> () {
  // CHECK: tosa.equal %arg0, %arg1 : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  %0 = tosa.equal %arg0, %arg1 : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: test_non_tosa_consumer_shape
func.func @test_non_tosa_consumer_shape(%arg0: tensor<4x4xf32>) -> !shape.shape {
  // CHECK: tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<*xf32>
  %1 = shape.shape_of %0 : tensor<*xf32> -> !shape.shape
  return %1 : !shape.shape
}

// -----

// CHECK-LABEL: test_non_tosa_consumer_shape
func.func @test_non_tosa_consumer_shape2(%arg0: tensor<4x4xf32>) -> tensor<?xindex> {
  // CHECK: tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<*xf32>
  %1 = shape.shape_of %0 : tensor<*xf32> -> tensor<?xindex>
  return %1 : tensor<?xindex>
}

// -----

// CHECK-LABEL: test_non_tosa_consumer_extract
func.func @test_non_tosa_consumer_extract(%arg0: tensor<4x4xf32>, %arg1: index) -> f32 {
  // CHECK: tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %0 = tosa.log %arg0 : (tensor<4x4xf32>) -> tensor<?x?xf32>
  %1 = tensor.extract %0[%arg1, %arg1] : tensor<?x?xf32>
  return %1 : f32
}

// -----

// CHECK-LABEL: test_non_tosa_consumer_still_propagates
func.func @test_non_tosa_consumer_still_propagates(%arg0: tensor<1x1x8xf32>, %arg1: tensor<1x8x1xf32>) -> tensor<?x?xf32> {
  // CHECK: tosa.matmul %arg0, %arg1 : (tensor<1x1x8xf32>, tensor<1x8x1xf32>) -> tensor<1x1x1xf32>
  %0 = tosa.matmul %arg0, %arg1 : (tensor<1x1x8xf32>, tensor<1x8x1xf32>) -> tensor<?x1x1xf32>
  %1 = arith.constant dense<[1, 1]> : tensor<2xindex>
  %2 = tensor.reshape %0(%1) : (tensor<?x1x1xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: test_tosa_use_def_chain
func.func @test_tosa_use_def_chain(%arg0: tensor<1x32x32x3xf32>, %arg1: tensor<16x3x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<?x16x16x16xf32> {
  // CHECK: [[CONV:%.+]] = tosa.conv2d %arg0, %arg1, %arg2
  // CHECK: (tensor<1x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x32x32x16xf32>
  %0 = tosa.conv2d %arg0, %arg1, %arg2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x3xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<?x32x32x16xf32>
  // CHECK: tosa.max_pool2d [[CONV]]
  // CHECK: (tensor<1x32x32x16xf32>) -> tensor<1x16x16x16xf32>
  %1 = tosa.max_pool2d %0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<?x32x32x16xf32>) -> tensor<?x16x16x16xf32>
  return %1 : tensor<?x16x16x16xf32>
}

// -----

// This test locks two bug fixes manifested in the code below.
//
// 1. Context
//
// When shape propagation hits an operation that does not support shape
// inference (here 'tensor.expand_shape'), it must revert the currently
// inferred shape of its consumers back to the originally expected input
// type to avoid potential op verification errors. This type reversal is
// done through an additional 'tensor.cast' op.
//
//
// 2. Preserving list of non-inferrable consumers
//
// When multiple non-inferrable consumers of a shape-inferred value are found
// (here, the 2 occurrences of 'tensor.expand_shape' consuming the output of
// 'tosa.cast'), their input argument ('%0') must be altered to consume the
// output the new 'tensor.cast' op. While these replacements occur, the use list
// of the producer ('tosa.cast') is also implicitly altered, invalidating any
// iterators associated with it. It is therefore necessary to create a copy of
// this use list ahead of time. Before this bug fix, the second
// 'tensor.expand_shape' op below was not updated correctly.
//
// 3. Guaranteeing def-use order
//
// When emitting the 'tensor.cast' op, it is important to guarantee that its
// output value is defined before all of its consumers (here, both of the
// 'tensor.expand_shape' ops. In a previous version of the code, this insertion
// occurred right before the first encountered consumer. Since use lists are
// saved in reverse order, the 'tensor.cast' op was inserted before the second
// 'tensor.expand_shape' op, leading to a def-use order violation when the
// first 'tensor.expand_shape' op was later updated. The current implementation
// sets the insertion point right after the producer of the last shape-inferred
// value (here 'tosa.cast'), which guarantees correct def-use order for all
// future operand updates.

// CHECK-LABEL: test_multiple_non_inferrable_consumers
// CHECK-SAME: %[[ARG:.*]]: tensor<1x2x8xf32>
func.func @test_multiple_non_inferrable_consumers(%arg0: tensor<1x2x8xf32>) {
  // CHECK: %[[TOSA_CAST:.*]] = tosa.cast %[[ARG]] : (tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
  // CHECK: %[[TENSOR_CAST:.*]] = tensor.cast %[[TOSA_CAST]] : tensor<1x2x8xf32> to tensor<?x2x8xf32>
  %0 = tosa.cast %arg0 : (tensor<1x2x8xf32>) -> tensor<?x2x8xf32>

  %c0 = arith.constant 0 : index
  %dim = tensor.dim %0, %c0 : tensor<?x2x8xf32>

  // CHECK: tensor.expand_shape %[[TENSOR_CAST]]
  // CHECK: tensor.expand_shape %[[TENSOR_CAST]]
  %expanded_0 = tensor.expand_shape %0 [[0], [1, 2], [3]] output_shape [%dim, 1, 4, 8] : tensor<?x2x8xf32> into tensor<?x1x2x8xf32>
  %expanded_1 = tensor.expand_shape %0 [[0], [1, 2], [3]] output_shape [%dim, 1, 4, 8] : tensor<?x2x8xf32> into tensor<?x1x2x8xf32>
  return
}

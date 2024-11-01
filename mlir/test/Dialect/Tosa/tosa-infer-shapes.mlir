// RUN: mlir-opt --split-input-file --tosa-infer-shapes %s | FileCheck %s

// CHECK-LABEL: @test_return
func.func @test_return(%arg0 : tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: [[LOG:%.+]] = "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tensor.cast [[LOG]] : tensor<4xf32> to tensor<*xf32>
  %0 = "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_multiple
func.func @test_multiple(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  // CHECK: [[ADD:%.+]] = "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: [[LOG:%.+]] = "tosa.log"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "tosa.log"(%0) : (tensor<*xf32>) -> tensor<*xf32>

  // CHECK: [[SUB:%.+]] = "tosa.sub"(%0, %arg2) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %2 = "tosa.sub"(%0, %arg2) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_unary_f32
func.func @test_unary_f32(%arg0 : tensor<4xf32>) -> () {
  // CHECK: "tosa.abs"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.ceil"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1 = "tosa.ceil"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.clamp"(%arg0) {{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %2 = "tosa.clamp"(%arg0) { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.exp"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %3 = "tosa.exp"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.floor"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %4 = "tosa.floor"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %5 = "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.negate"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %6 = "tosa.negate"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.reciprocal"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %7 = "tosa.reciprocal"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
  %8 = "tosa.reverse"(%arg0) { axis = 0 : i64 } : (tensor<4xf32>) -> tensor<?xf32>

  // CHECK: "tosa.rsqrt"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %9 = "tosa.rsqrt"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.tanh"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %10 = "tosa.tanh"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.sigmoid"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %11 = "tosa.sigmoid"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.cast"(%arg0) : (tensor<4xf32>) -> tensor<4xi32>
  %12 = "tosa.cast"(%arg0) : (tensor<4xf32>) -> tensor<*xi32>
  return
}

// -----

// CHECK-LABEL: @test_unary_i32
func.func @test_unary_i32(%arg0 : tensor<4xi32>) -> () {
  // CHECK: "tosa.abs"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %0 = "tosa.abs"(%arg0) : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.bitwise_not"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %1 = "tosa.bitwise_not"(%arg0) : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.clamp"(%arg0) {{.+}} : (tensor<4xi32>) -> tensor<4xi32>
  %2 = "tosa.clamp"(%arg0) { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.clz"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %3 = "tosa.clz"(%arg0) : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.negate"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %4 = "tosa.negate"(%arg0) : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<4xi32>) -> tensor<4xi32>
  %5 = "tosa.reverse"(%arg0) { axis = 0 : i64 } : (tensor<4xi32>) -> tensor<?xi32>

  // CHECK: "tosa.rescale"(%arg0) {{.+}} : (tensor<4xi32>) -> tensor<4xi16>
  %6 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [42 : i32, 43 : i32], shift = [14 : i32, 15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<4xi32>)  -> (tensor<*xi16>)

  // CHECK: "tosa.identity"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %7 = "tosa.identity"(%arg0) : (tensor<4xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_binary_scalar_f32
func.func @test_binary_scalar_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<f32>) -> () {
  // CHECK: "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = "tosa.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %2 = "tosa.minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %3 = "tosa.mul"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %4 = "tosa.pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.sub"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %5 = "tosa.sub"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xf32>

  // CHECK: "tosa.equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xi1>
  %6 = "tosa.equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>

  // CHECK: "tosa.greater"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xi1>
  %7 = "tosa.greater"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>

  // CHECK: "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<4xi1>
  %8 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<f32>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_binary_broadcast_f32
func.func @test_binary_broadcast_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>) -> () {
  // CHECK: "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %1 = "tosa.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %2 = "tosa.minimum"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %3 = "tosa.mul"(%arg0, %arg1) { shift = 0 : i32 } : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %4 = "tosa.pow"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.sub"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xf32>
  %5 = "tosa.sub"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xf32>

  // CHECK: "tosa.equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %6 = "tosa.equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: "tosa.greater"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %7 = "tosa.greater"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  // CHECK: "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<4xi1>
  %8 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xf32>, tensor<1xf32>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_binary_i32
func.func @test_binary_i32(%arg0 : tensor<4xi32>, %arg1 : tensor<i32>) -> () {
  // CHECK: "tosa.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.bitwise_and"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %1 = "tosa.bitwise_and"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.bitwise_or"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %2 = "tosa.bitwise_or"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.bitwise_xor"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %3 = "tosa.bitwise_xor"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.equal"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi1>
  %4 = "tosa.equal"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>

  // CHECK: "tosa.greater"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi1>
  %5 = "tosa.greater"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>

  // CHECK: "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi1>
  %6 = "tosa.greater_equal"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi1>

  // CHECK: "tosa.logical_left_shift"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %7 = "tosa.logical_left_shift"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.logical_right_shift"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %8 = "tosa.logical_right_shift"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.maximum"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %9 = "tosa.maximum"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.minimum"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %10 = "tosa.minimum"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.mul"(%arg0, %arg1) {shift = 0 : i32} : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %11 = "tosa.mul"(%arg0, %arg1) { shift = 0 : i32 }: (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK: "tosa.pow"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %12 = "tosa.pow"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  // CHECK:  "tosa.sub"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<4xi32>
  %13 = "tosa.sub"(%arg0, %arg1) : (tensor<4xi32>, tensor<i32>) -> tensor<*xi32>

  return
}

// -----

// CHECK-LABEL: @test_binary_i1
func.func @test_binary_i1(%arg0 : tensor<4xi1>, %arg1 : tensor<i1>) -> () {
  // CHECK "tosa.logical_and"(%arg0, %arg1) : (tensor<4xi1>, tensor<i1>) -> tensor<4xi1>
  %0 = "tosa.logical_and"(%arg0, %arg1): (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>

  // CHECK "tosa.logical_or"(%arg0, %arg1) : (tensor<4xi1>, tensor<i1>) -> tensor<4xi1>
  %1 = "tosa.logical_or"(%arg0, %arg1): (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>

  // CHECK "tosa.logical_xor"(%arg0, %arg1) : (tensor<4xi1>, tensor<i1>) -> tensor<*4i1>
  %2 = "tosa.logical_xor"(%arg0, %arg1): (tensor<4xi1>, tensor<i1>) -> tensor<*xi1>

  return
}

// -----

// CHECK-LABEL: @test_select_i32
func.func @test_select_i32(%arg0 : tensor<4xi1>, %arg1 : tensor<i32>, %arg2 : tensor<4xi32>) -> () {
  // CHECK: "tosa.select"(%arg0, %arg1, %arg2) : (tensor<4xi1>, tensor<i32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = "tosa.select"(%arg0, %arg1, %arg2): (tensor<4xi1>, tensor<i32>, tensor<4xi32>) -> tensor<*xi32>

  return
}

// -----

// CHECK-LABEL: @test_static_argmax
func.func @test_static_argmax(%arg0 : tensor<2x3xi32>) -> () {
  // CHECK: "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x3xi32>) -> tensor<3xi32>
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x3xi32>) -> tensor<?xi32>

  // CHECK: "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x3xi32>) -> tensor<2xi32>
  %1 = "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x3xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_dynamic_argmax
func.func @test_dynamic_argmax(%arg0 : tensor<2x?xi32>) -> () {
  // CHECK: "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>

  // CHECK: "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x?xi32>) -> tensor<2xi32>
  %1 = "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_static_fully_connected
func.func @test_static_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_input_fully_connected
func.func @test_static_input_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<3x?xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_weight_fully_connected
func.func @test_static_weight_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_bias_fully_connected
func.func @test_static_bias_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_out_fully_connected
func.func @test_static_out_fully_connected(%arg0 : tensor<3x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_matmul
func.func @test_static_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_lhs_matmul
func.func @test_dynamic_lhs_matmul(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<2x?x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_rhs_matmul
func.func @test_dynamic_rhs_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<?x?x?xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<2x3x?xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_mixed_matmul
func.func @test_dynamic_mixed_matmul(%arg0 : tensor<?x3x?xi32>, %arg1 : tensor<?x?x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x3x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABLE: @test_table_static
func.func @test_table_static(%arg0 : tensor<4x5xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:"tosa.table"(%arg0, %arg1) : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<4x5xi16>
  %0 = "tosa.table"(%arg0, %arg1) : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABLE: @test_table_dynamic
func.func @test_table_dynamic(%arg0 : tensor<4x?xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:"tosa.table"(%arg0, %arg1) : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<4x?xi16>
  %0 = "tosa.table"(%arg0, %arg1) : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABEL: @test_static_reshape
func.func @test_static_reshape(%arg0 : tensor<4x4xi32>) -> () {
  // CHECK: "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x4xi32>) -> tensor<16xi32>
  %0 = "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x4xi32>)  -> tensor<?xi32>

  // CHECK: "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x4xi32>) -> tensor<16xi32>
  %1 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x4xi32>)  -> tensor<?xi32>

  // CHECK: "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x4xi32>) -> tensor<2x8xi32>
  %2 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x4xi32>)  -> tensor<?x?xi32>

  return
}
// -----

// CHECK-LABEL: @test_dynamic_reshape
func.func @test_dynamic_reshape(%arg0 : tensor<4x?xi32>) -> () {
  // CHECK: %0 = "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x?xi32>) -> tensor<16xi32>
  %0 = "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x?xi32>)  -> tensor<?xi32>

  // CHECK: %1 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x?xi32>) -> tensor<?xi32>
  %1 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x?xi32>)  -> tensor<?xi32>

  // CHECK: %2 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x?xi32>) -> tensor<2x?xi32>
  %2 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x?xi32>)  -> tensor<?x?xi32>

  return
}

// -----

// CHECK: @test_reduce_binary
func.func @test_reduce_binary(%arg0 : tensor<2x3x?x?xi1>) -> () {
  // CHECK: "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xi1>) -> tensor<1x3x?x?xi1>
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: "tosa.reduce_all"(%arg0) {axis = 1 : i64} : (tensor<2x3x?x?xi1>) -> tensor<2x1x?x?xi1>
  %1 = "tosa.reduce_all"(%arg0) {axis = 1 : i64} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: "tosa.reduce_all"(%arg0) {axis = 2 : i64} : (tensor<2x3x?x?xi1>) -> tensor<2x3x1x?xi1>
  %2 = "tosa.reduce_all"(%arg0) {axis = 2 : i64} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: "tosa.reduce_all"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xi1>) -> tensor<2x3x?x1xi1>
  %3 = "tosa.reduce_all"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  // CHECK: "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xi1>) -> tensor<1x3x?x?xi1>
  %4 = "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xi1>) -> tensor<?x?x?x?xi1>

  return
}

// -----

// CHECK: @test_reduce_float
func.func @test_reduce_float(%arg0 : tensor<2x3x?x?xf32>) -> () {
  // CHECK: "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xf32>) -> tensor<1x3x?x?xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x1x?x?xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_sum"(%arg0) {axis = 2 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x3x1x?xf32>
  %2 = "tosa.reduce_sum"(%arg0) {axis = 2 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_sum"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %3 = "tosa.reduce_sum"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_max"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %4 = "tosa.reduce_max"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_min"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %5 = "tosa.reduce_min"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: "tosa.reduce_prod"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<2x3x?x1xf32>
  %6 = "tosa.reduce_prod"(%arg0) {axis = 3 : i64} : (tensor<2x3x?x?xf32>) -> tensor<?x?x?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat
func.func @test_concat(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic
func.func @test_concat_dynamic(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x?xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<3x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic_axis
func.func @test_concat_dynamic_axis(%arg0 : tensor<?x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_axis_1
func.func @test_concat_axis_1(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_failure
func.func @test_concat_failure(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_padding_no_const
func.func @test_padding_no_const(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xi32>) -> () {
  // CHECK: "tosa.pad"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  %0 = "tosa.pad"(%arg0, %arg1)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL:@test_padding_dynamic_input
func.func @test_padding_dynamic_input(%arg0 : tensor<1x?xf32>) -> () {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: "tosa.pad"(%arg0, %cst) : (tensor<1x?xf32>, tensor<2x2xi32>) -> tensor<4x?xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x?xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_padding_simple
func.func @test_padding_simple(%arg0 : tensor<1x2xf32>) -> () {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: "tosa.pad"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_slice
func.func @test_slice(%arg0 : tensor<?xi32>) -> () {
  // CHECK: "tosa.slice"(%arg0) {size = [2], start = [1]} : (tensor<?xi32>) -> tensor<2xi32>
  %0 = "tosa.slice"(%arg0) { size = [2], start = [1] } : (tensor<?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_slice_dynamic
func.func @test_slice_dynamic(%arg0 : tensor<10x?x2xf32>) -> () {
  // CHECK: "tosa.slice"(%arg0) {size = [7, -1, 1], start = [1, 0, 0]} : (tensor<10x?x2xf32>) -> tensor<7x?x1xf32>
  %0 = "tosa.slice"(%arg0) {size = [7, -1, 1], start = [1, 0, 0]} : (tensor<10x?x2xf32>) -> tensor<?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_tile
func.func @test_tile(%arg0 : tensor<2x3x?xi32>) -> () {
  // CHECK: "tosa.tile"(%arg0) {multiples = [2, 1, 5]} : (tensor<2x3x?xi32>) -> tensor<4x3x?xi32>
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1, 5]} : (tensor<2x3x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_same
func.func @test_transpose_same(%arg0 : tensor<4x4x4xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x4xi32>, tensor<3xi32>) -> tensor<4x4x4xi32>
  %0 = "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x4xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_perm_unknown
func.func @test_transpose_perm_unknown(%arg0 : tensor<4x4x5xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %0 = "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_static
func.func @test_transpose_static(%arg0 : tensor<3x4x5xi32>) -> () {
  %0 = arith.constant dense<[2, 1, 0]> : tensor<3xi32>
  // CHECK: "tosa.transpose"(%arg0, %cst) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<5x4x3xi32>
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_static
func.func @gather_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x6xi32>) -> tensor<3x6x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_values
func.func @gather_dynamic_values(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<3x6xi32>) -> tensor<3x6x?xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<3x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_indices
func.func @gather_dynamic_indices(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<?x?xi32>) -> tensor<3x?x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_minimum_info
func.func @gather_minimum_info(%arg0 : tensor<3x?x5xi32>, %arg1 : tensor<?x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x?x5xi32>, tensor<?x6xi32>) -> tensor<3x6x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x?x5xi32>, tensor<?x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static
func.func @scatter_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_values
func.func @scatter_static_values(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_indices
func.func @scatter_static_indices(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>) -> tensor<3x?x?xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_input
func.func @scatter_static_input(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>) -> tensor<3x?x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_minimum_static
func.func @scatter_minimum_static(%arg0 : tensor<?x4x?xi32>, %arg1 : tensor<3x?xi32>, %arg2 : tensor<?x?x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_pool_static
func.func @test_pool_static(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x2x4x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x2x4x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_static
func.func @conv2d_static(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_input
func.func @conv2d_dynamic_input(%input: tensor<?x?x?x?xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<?x?x?x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_pool_dynamic_input
func.func @test_pool_dynamic_input(%arg0: tensor<?x?x?x?xf32>) {
  // CHECK: -> tensor<?x?x?x?xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<?x?x?x?xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_padded
func.func @test_pool_padded(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x5x11x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [1, 2, 3, 4], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x5x11x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [1, 2, 3, 4], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_weight
func.func @conv2d_dynamic_weight(%input: tensor<2x8x9x3xf32>, %weights: tensor<?x?x?x?xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x?x?x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_dynamic_bias
func.func @conv2d_dynamic_bias(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<?xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_pool_stride
func.func @test_pool_stride(%arg0: tensor<3x11x12x7xf32>) {
  // CHECK: -> tensor<3x4x4x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [2, 3]} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x4x4x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [2, 3]} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv2d_padded
func.func @conv2d_padded(%input: tensor<2x8x9x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x9x11x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [1, 2, 3, 4], stride = [1, 1], dilation = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_dilated
func.func @conv2d_dilated(%input: tensor<2x12x14x3xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x5xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [3, 2]} : (tensor<2x12x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_strided
func.func @conv2d_strided(%input: tensor<1x13x14x1xf32>, %weights: tensor<1x1x1x1xf32>, %bias: tensor<1xf32>) -> () {
  // CHECK: -> tensor<1x5x7x1xf32>
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [3, 2], dilation = [1, 1]} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>)  -> (tensor<?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv3d_static
func.func @conv3d_static(%input: tensor<2x8x9x10x3xf32>, %weights: tensor<5x3x6x4x3xf32>, %bias: tensor<5xf32>) -> () {
  // CHECK: -> tensor<2x6x4x7x5xf32>
  %0 = "tosa.conv3d"(%input, %weights, %bias) {dilation = [1, 1, 1], pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_input
func.func @conv3d_dynamic_input(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<?x?x?x?x5xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [1, 1, 1], pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} : (tensor<?x?x?x?x?xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_weight
func.func @conv3d_dynamic_weight(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<?x?x?x?x?xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x?x?x?x5xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [1, 1, 1], pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} : (tensor<2x8x9x10x3xf32>, tensor<?x?x?x?x?xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dynamic_bias
func.func @conv3d_dynamic_bias(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x6x4x7x5xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [1, 1, 1], pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_padded
func.func @conv3d_padded(%arg0: tensor<2x8x9x10x3xf32>, %arg1: tensor<5x3x6x4x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x9x11x18x5xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [1, 1, 1], pad = [1, 2, 3, 4, 5, 6], stride = [1, 1, 1]} : (tensor<2x8x9x10x3xf32>, tensor<5x3x6x4x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_dilated
func.func @conv3d_dilated(%arg0: tensor<2x12x14x16x3xf32>, %arg1: tensor<5x3x6x2x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x6x4x12x5xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [3, 2, 4], pad = [0, 0, 0, 0, 0, 0], stride = [1, 1, 1]} : (tensor<2x12x14x16x3xf32>, tensor<5x3x6x2x3xf32>, tensor<5xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @conv3d_strided
func.func @conv3d_strided(%arg0: tensor<1x13x14x15x1xf32>, %arg1: tensor<1x1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x5x7x4x1xf32>
  %0 = "tosa.conv3d"(%arg0, %arg1, %arg2) {dilation = [1, 1, 1], pad = [0, 0, 0, 0, 0, 0], stride = [3, 2, 4]} : (tensor<1x13x14x15x1xf32>, tensor<1x1x1x1x1xf32>, tensor<1xf32>) -> tensor<?x?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_static
func.func @depthwise_conv2d_static(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_input
func.func @depthwise_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<?x?x?x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<?x?x?x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_weight
func.func @depthwise_conv2d_dynamic_weight(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x?x?x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<15xf32>) -> tensor<2x?x?x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dynamic_bias
func.func @depthwise_conv2d_dynamic_bias(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<?xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_padded
func.func @depthwise_conv2d_padded(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x9x11x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [1, 2, 3, 4], stride = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x9x11x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_dilated
func.func @depthwise_conv2d_dilated(%arg0: tensor<2x12x14x3xf32>, %arg1: tensor<3x6x3x5xf32>, %arg2: tensor<15xf32>) {
  // CHECK: -> tensor<2x6x4x15xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [3, 2], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<2x12x14x3xf32>, tensor<3x6x3x5xf32>, tensor<15xf32>) -> tensor<2x6x4x15xf32>
  return
}

// -----

// CHECK-LABEL: @depthwise_conv2d_strided
func.func @depthwise_conv2d_strided(%arg0: tensor<1x13x14x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x5x7x1xf32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [3, 2]} : (tensor<1x13x14x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x5x7x1xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_out_shape
func.func @transpose_conv2d_out_shape(%arg0: tensor<2x?x?x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x8x9x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, 8, 9, -1], stride = [1, 1]} : (tensor<2x?x?x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x8x9x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_static
func.func @transpose_conv2d_static(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x18x19x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_static_strided
func.func @transpose_conv2d_static_strided(%arg0: tensor<2x16x14x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x33x45x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [2, 3]} : (tensor<2x16x14x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_input
func.func @transpose_conv2d_dynamic_input(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<?x?x?x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_weights
func.func @transpose_conv2d_dynamic_weights(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x?x?x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x6x4x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_dynamic_bias
func.func @transpose_conv2d_dynamic_bias(%arg0: tensor<2x6x4x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<?xf32>) {
  // CHECK: -> tensor<2x8x9x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x6x4x3xf32>, tensor<5x3x6x3xf32>, tensor<?xf32>) -> tensor<2x8x9x5xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv2d_padded
func.func @transpose_conv2d_padded(%arg0: tensor<2x9x11x3xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
  // CHECK: -> tensor<2x10x13x5xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [1, 0, 3, 0], out_shape = [-1, -1, -1, -1], stride = [1, 1]} : (tensor<2x9x11x3xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<2x10x13x5xf32>
  return
}

// CHECK-LABEL: @transpose_conv2d_strided
func.func @transpose_conv2d_strided(%arg0: tensor<1x5x7x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1xf32>) {
  // CHECK: -> tensor<1x13x13x1xf32>
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {out_pad = [0, 0, 0, 0], out_shape = [-1, -1, -1, -1], stride = [3, 2]} : (tensor<1x5x7x1xf32>, tensor<1x1x1x1xf32>, tensor<1xf32>) -> tensor<1x13x13x1xf32>
  return
}

// -----

// CHECK-LABEL: @resize_int_horizontal
func.func @resize_int_horizontal(%arg0: tensor<1x15x13x1xi8>) {
  // CHECK: -> tensor<1x23x179x1xi8>
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [11, 7, 89, 6], offset = [0, 0], border = [0, 0]} : (tensor<1x15x13x1xi8>) -> tensor<?x?x?x?xi8>
  return
}

// -----

// CHECK-LABEL: @resize_int_vertical
func.func @resize_int_vertical(%arg0: tensor<1x49x42x1xi16>) {
  // CHECK: -> tensor<1x112x220x1xi16>
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [37, 16, 219, 41], offset = [0, 0], border = [0, 0]} : (tensor<1x49x42x1xi16>) -> tensor<?x?x?x?xi16>
  return
}

// -----

// CHECK-LABEL: @resize_int_power_of_two_upscale
func.func @resize_int_power_of_two_upscale(%arg0: tensor<1x23x19x1xi8>) {
  // CHECK: -> tensor<1x353x289x1xi32>
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [16, 1, 16, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x23x19x1xi8>) -> tensor<?x?x?x?xi32>
  return
}

// -----

// CHECK-LABEL: @resize_int_power_of_two_upscale_offsetted
func.func @resize_int_power_of_two_upscale_offsetted(%arg0: tensor<1x41x26x1xi16>) {
  // CHECK: -> tensor<1x328x208x1xi48>
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [16, 2, 16, 2], offset = [-7, -7], border = [7, 7]} : (tensor<1x41x26x1xi16>) -> tensor<?x?x?x?xi48>
  return
}

// -----
// CHECK-LABEL: @resize_fp_horizontal
func.func @resize_fp_horizontal(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x106x85x1xf32>
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [15, 7, 84, 47], offset = [0, 0], border = [0, 0]} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----
// CHECK-LABEL: @resize_fp_vertical
func.func @resize_fp_vertical(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x128x13x1xf32>
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [127, 49, 12, 47], offset = [0, 0], border = [0, 0]} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @resize_fp_power_of_two_upscale
func.func @resize_fp_power_of_two_upscale(%arg0: tensor<1x23x23x1xf32>) {
  // CHECK: -> tensor<1x89x89x1xf32>
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [4, 1, 4, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x23x23x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @resize_fp_power_of_two_upscale_offsetted
func.func @resize_fp_power_of_two_upscale_offsetted(%arg0: tensor<1x50x48x1xf32>) {
  // CHECK: -> tensor<1x1600x1536x1xf32>
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [64, 2, 64, 2], offset = [-31, -31], border = [31, 31]} : (tensor<1x50x48x1xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @if_test_simple
func.func @if_test_simple(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> () {
  %a = "tosa.log"(%arg0) : (tensor<f32>) -> tensor<*xf32>
  %b = "tosa.log"(%arg1) : (tensor<f32>) -> tensor<*xf32>
  // CHECK: (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "tosa.cond_if"(%arg2, %a, %b) ({
  ^bb1(%arg3 : tensor<*xf32>, %arg4 : tensor<*xf32>):
    "tosa.yield"(%arg3) : (tensor<*xf32>) -> ()
  }, {
  ^bb1(%arg5 : tensor<*xf32>, %arg6 : tensor<*xf32>):
    "tosa.yield"(%arg6) : (tensor<*xf32>) -> ()
  }) : (tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: @if_test_dynamic
func.func @if_test_dynamic(%arg0 : tensor<2xf32>, %arg1 : tensor<3xf32>, %arg2 : tensor<i1>) -> () {
  // CHECK: (tensor<i1>, tensor<2xf32>, tensor<3xf32>) -> tensor<?xf32>
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb1(%arg3 : tensor<2xf32>, %arg4 : tensor<3xf32>):
    "tosa.yield"(%arg3) : (tensor<2xf32>) -> ()
  }, {
  ^bb1(%arg5 : tensor<2xf32>, %arg6 : tensor<3xf32>):
    "tosa.yield"(%arg6) : (tensor<3xf32>) -> ()
  }) : (tensor<i1>, tensor<2xf32>, tensor<3xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: @if_test_unranked
func.func @if_test_unranked(%arg0 : tensor<f32>, %arg1 : tensor<3xf32>, %arg2 : tensor<i1>) -> () {
  // CHECK: (tensor<i1>, tensor<f32>, tensor<3xf32>) -> tensor<*xf32>
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb1(%arg3 : tensor<f32>, %arg4 : tensor<3xf32>):
    "tosa.yield"(%arg3) : (tensor<f32>) -> ()
  }, {
  ^bb1(%arg5 : tensor<f32>, %arg6 : tensor<3xf32>):
    "tosa.yield"(%arg6) : (tensor<3xf32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<3xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: @if_test_propagate
func.func @if_test_propagate(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<i1>) -> () {
  // CHECK: (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "tosa.cond_if"(%arg2, %arg0, %arg1) ({
  ^bb1(%arg3 : tensor<*xf32>, %arg4 : tensor<*xf32>):
    %1 = "tosa.add"(%arg3, %arg4) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tosa.yield"(%1) : (tensor<*xf32>) -> ()
  }, {
  ^bb1(%arg5 : tensor<*xf32>, %arg6 : tensor<*xf32>):
    %1 = "tosa.sub"(%arg5, %arg6) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tosa.yield"(%1) : (tensor<*xf32>) -> ()
  }) : (tensor<i1>, tensor<f32>, tensor<f32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: @while_test
func.func @while_test(%arg0 : tensor<i32>) -> (tensor<*xi32>) {
  // CHECK:      "tosa.add"
  // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<*xi32>

  // CHECK:      "tosa.while_loop"
  %1 = "tosa.while_loop"(%0) ({

  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  ^bb0(%arg2: tensor<*xi32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
    // CHECK:       "tosa.greater_equal"
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.greater_equal"(%2, %arg2) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
    // CHECK:      "tosa.yield"
    // CHECK-SAME: tensor<i1>
    "tosa.yield"(%3) : (tensor<*xi1>) -> ()
  },  {
  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  ^bb0(%arg2: tensor<*xi32>):
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

    // CHECK:     "tosa.add"
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tosa.add"(%arg2, %2) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>

    // CHECK:      "tosa.yield"
    // CHECK-SAME: tensor<i32>
    "tosa.yield"(%3) : (tensor<*xi32>) -> ()

  // CHECK:      (tensor<i32>) -> tensor<i32>
  }) : (tensor<*xi32>) -> (tensor<*xi32>)

  // CHECK:      tensor.cast
  return %1 : tensor<*xi32>
}

// -----

// CHECK-LABEL: @while_test
func.func @while_test(%arg0 : tensor<i32>, %arg1 : tensor<1xi32>) -> () {
  // CHECK:      "tosa.while_loop"
  %1:2 = "tosa.while_loop"(%arg0, %arg1) ({

  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  // CHECK-SAME: tensor<?xi32>
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xi32>):
    %2 = "tosa.const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>

    // CHECK:       "tosa.greater_equal"
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = "tosa.greater_equal"(%2, %arg2) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi1>
    "tosa.yield"(%3) : (tensor<*xi1>) -> ()
  },  {

  // CHECK:      ^bb0
  // CHECK-SAME: tensor<i32>
  // CHECK-SAME: tensor<?xi32>
  ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xi32>):
    %2 = "tosa.const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>

    // CHECK:     "tosa.add"
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "tosa.add"(%arg2, %2) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>

    // CHECK:      "tosa.concat"
    // CHECK-SAME: (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %4 = "tosa.concat"(%arg3, %arg3) { axis = 0 : i64 } : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)

    // CHECK:      "tosa.yield"
    // CHECK-SAME: tensor<i32>
    // CHECK-SAME: tensor<?xi32>
    "tosa.yield"(%3, %4) : (tensor<*xi32>, tensor<*xi32>) -> ()

  // CHECK:      (tensor<i32>, tensor<1xi32>) -> (tensor<i32>, tensor<?xi32>)
  }) : (tensor<i32>, tensor<1xi32>) -> (tensor<*xi32>, tensor<*xi32>)
  return
}

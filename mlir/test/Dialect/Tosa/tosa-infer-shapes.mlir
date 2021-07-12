// RUN: mlir-opt --split-input-file --tosa-infer-shapes %s | FileCheck %s

// CHECK-LABEL: @test_return
func @test_return(%arg0 : tensor<4xf32>) -> tensor<*xf32> {
  // CHECK: [[LOG:%.+]] = "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK: tensor.cast [[LOG]] : tensor<4xf32> to tensor<*xf32>
  %0 = "tosa.log"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_multiple
func @test_multiple(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
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
func @test_unary_f32(%arg0 : tensor<4xf32>) -> () {
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

  // CHECK: "tosa.reluN"(%arg0) {{.+}} : (tensor<4xf32>) -> tensor<4xf32>
  %8 = "tosa.reluN"(%arg0) { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
  %9 = "tosa.reverse"(%arg0) { axis = 0 : i64 } : (tensor<4xf32>) -> tensor<?xf32>

  // CHECK: "tosa.rsqrt"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %10 = "tosa.rsqrt"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.tanh"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %11 = "tosa.tanh"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>

  // CHECK: "tosa.sigmoid"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %12 = "tosa.sigmoid"(%arg0) : (tensor<4xf32>) -> tensor<*xf32>
  return
}

// -----

// CHECK-LABEL: @test_unary_i32
func @test_unary_i32(%arg0 : tensor<4xi32>) -> () {
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

  // CHECK: "tosa.reluN"(%arg0) {{.+}} : (tensor<4xi32>) -> tensor<4xi32>
  %5 = "tosa.reluN"(%arg0) { max_int = 10 : i64, min_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 10.0 : f32 } : (tensor<4xi32>) -> tensor<*xi32>

  // CHECK: "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<4xi32>) -> tensor<4xi32>
  %6 = "tosa.reverse"(%arg0) { axis = 0 : i64 } : (tensor<4xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_unary_i1
func @test_unary_i1(%arg0 : tensor<4xi1>) -> () {
  // CHECK: "tosa.logical_not"(%arg0) : (tensor<4xi1>) -> tensor<4xi1>
  %0 = "tosa.logical_not"(%arg0) : (tensor<4xi1>) -> tensor<*xi1>
  return
}

// -----

// CHECK-LABEL: @test_binary_scalar_f32
func @test_binary_scalar_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<f32>) -> () {
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
func @test_binary_broadcast_f32(%arg0 : tensor<4xf32>, %arg1 : tensor<1xf32>) -> () {
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
func @test_binary_i32(%arg0 : tensor<4xi32>, %arg1 : tensor<i32>) -> () {
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
func @test_binary_i1(%arg0 : tensor<4xi1>, %arg1 : tensor<i1>) -> () {
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
func @test_select_i32(%arg0 : tensor<4xi1>, %arg1 : tensor<i32>, %arg2 : tensor<4xi32>) -> () {
  // CHECK: "tosa.select"(%arg0, %arg1, %arg2) : (tensor<4xi1>, tensor<i32>, tensor<4xi32>) -> tensor<4xi32>
  %0 = "tosa.select"(%arg0, %arg1, %arg2): (tensor<4xi1>, tensor<i32>, tensor<4xi32>) -> tensor<*xi32>

  return
}

// -----

// CHECK-LABEL: @test_static_argmax
func @test_static_argmax(%arg0 : tensor<2x3xi32>) -> () {
  // CHECK: "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x3xi32>) -> tensor<3xi32>
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x3xi32>) -> tensor<?xi32>

  // CHECK: "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x3xi32>) -> tensor<2xi32>
  %1 = "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x3xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_dynamic_argmax
func @test_dynamic_argmax(%arg0 : tensor<2x?xi32>) -> () {
  // CHECK: "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>

  // CHECK: "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x?xi32>) -> tensor<2xi32>
  %1 = "tosa.argmax"(%arg0) {axis = 1 : i64} : (tensor<2x?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_static_fully_connected
func @test_static_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<5x4xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_input_fully_connected
func @test_static_input_fully_connected(%arg0 : tensor<3x4xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<3x?xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x4xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_weight_fully_connected
func @test_static_weight_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<5x4xf32>, %arg2 : tensor<?xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<5x4xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_bias_fully_connected
func @test_static_bias_fully_connected(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_out_fully_connected
func @test_static_out_fully_connected(%arg0 : tensor<3x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<5xf32>) -> () {
  // CHECK: "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<3x5xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<3x?xf32>, tensor<?x?xf32>, tensor<5xf32>) -> tensor<?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_static_matmul
func @test_static_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<2x3x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_lhs_matmul
func @test_dynamic_lhs_matmul(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<2x4x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<2x?x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<2x4x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_rhs_matmul
func @test_dynamic_rhs_matmul(%arg0 : tensor<2x3x4xi32>, %arg1 : tensor<?x?x?xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<2x3x?xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<2x3x4xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABEL: @test_dynamic_mixed_matmul
func @test_dynamic_mixed_matmul(%arg0 : tensor<?x3x?xi32>, %arg1 : tensor<?x?x5xi32>) -> () {
  // CHECK: "tosa.matmul"(%arg0, %arg1) : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x3x5xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x3x?xi32>, tensor<?x?x5xi32>) -> tensor<?x?x?xi32>

  return
}

// -----

// CHECK-LABLE: @test_table_static
func @test_table_static(%arg0 : tensor<4x5xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:"tosa.table"(%arg0, %arg1) : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<4x5xi16>
  %0 = "tosa.table"(%arg0, %arg1) : (tensor<4x5xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABLE: @test_table_dynamic
func @test_table_dynamic(%arg0 : tensor<4x?xi16>, %arg1 : tensor<513xi16>) -> () {
  // CHECK:"tosa.table"(%arg0, %arg1) : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<4x?xi16>
  %0 = "tosa.table"(%arg0, %arg1) : (tensor<4x?xi16>, tensor<513xi16>) -> tensor<?x?xi16>
  return
}

// -----

// CHECK-LABEL: @test_static_reshape
func @test_static_reshape(%arg0 : tensor<4x4xi32>) -> () {
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
func @test_dynamic_reshape(%arg0 : tensor<4x?xi32>) -> () {
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
func @test_reduce_binary(%arg0 : tensor<2x3x?x?xi1>) -> () {
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
func @test_reduce_float(%arg0 : tensor<2x3x?x?xf32>) -> () {
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
func @test_concat(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<3x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic
func @test_concat_dynamic(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x?xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<3x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<1x2xf32>, tensor<2x?xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_dynamic_axis
func @test_concat_dynamic_axis(%arg0 : tensor<?x2xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<?x2xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_axis_1
func @test_concat_axis_1(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<2x3xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 1 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_concat_failure
func @test_concat_failure(%arg0 : tensor<2x1xf32>, %arg1 : tensor<2x2xf32>) -> () {
  // CHECK: "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>
  %0 = "tosa.concat"(%arg0, %arg1) {axis = 0 : i64} : (tensor<2x1xf32>, tensor<2x2xf32>) -> tensor<?x?xf32>

  return
}

// -----

// CHECK-LABEL: @test_padding_no_const
func @test_padding_no_const(%arg0 : tensor<1x2xf32>, %arg1 : tensor<2x2xi32>) -> () {
  // CHECK: "tosa.pad"(%arg0, %arg1) : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  %0 = "tosa.pad"(%arg0, %arg1)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL:@test_padding_dynamic_input
func @test_padding_dynamic_input(%arg0 : tensor<1x?xf32>) -> () {
  %0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: "tosa.pad"(%arg0, %cst) : (tensor<1x?xf32>, tensor<2x2xi32>) -> tensor<4x?xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x?xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_padding_simple
func @test_padding_simple(%arg0 : tensor<1x2xf32>) -> () {
  %0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: "tosa.pad"(%arg0, %cst) : (tensor<1x2xf32>, tensor<2x2xi32>) -> tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-LABEL: @test_slice
func @test_slice(%arg0 : tensor<?xi32>) -> () {
  // CHECK: "tosa.slice"(%arg0) {size = [2], start = [1]} : (tensor<?xi32>) -> tensor<2xi32>
  %0 = "tosa.slice"(%arg0) { size = [2], start = [1] } : (tensor<?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-LABEL: @test_tile
func @test_tile(%arg0 : tensor<2x3x?xi32>) -> () {
  // CHECK: "tosa.tile"(%arg0) {multiples = [2, 1, 5]} : (tensor<2x3x?xi32>) -> tensor<4x3x?xi32>
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1, 5]} : (tensor<2x3x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_same
func @test_transpose_same(%arg0 : tensor<4x4x4xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x4xi32>, tensor<3xi32>) -> tensor<4x4x4xi32>
  %0 = "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x4xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_perm_unknown
func @test_transpose_perm_unknown(%arg0 : tensor<4x4x5xi32>, %arg1 : tensor<3xi32>) -> () {
  // CHECK: "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x5xi32>, tensor<3xi32>) -> tensor<?x?x?xi32>
  %0 = "tosa.transpose"(%arg0, %arg1) : (tensor<4x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_transpose_static
func @test_transpose_static(%arg0 : tensor<3x4x5xi32>) -> () {
  %0 = constant dense<[2, 1, 0]> : tensor<3xi32>
  // CHECK: "tosa.transpose"(%arg0, %cst) : (tensor<3x4x5xi32>, tensor<3xi32>) -> tensor<5x4x3xi32>
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<3x4x5xi32>, tensor<3xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_static
func @gather_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x6xi32>) -> tensor<3x6x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_values
func @gather_dynamic_values(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<3x6xi32>) -> tensor<3x6x?xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<?x?x?xi32>, tensor<3x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_dynamic_indices
func @gather_dynamic_indices(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<?x?xi32>) -> tensor<3x?x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_minimum_info
func @gather_minimum_info(%arg0 : tensor<3x?x5xi32>, %arg1 : tensor<?x6xi32>) {
  // CHECK: "tosa.gather"(%arg0, %arg1) : (tensor<3x?x5xi32>, tensor<?x6xi32>) -> tensor<3x6x5xi32>
  %0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x?x5xi32>, tensor<?x6xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static
func @scatter_static(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<3x6xi32>, tensor<3x6x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_values
func @scatter_static_values(%arg0 : tensor<3x4x5xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<3x4x5xi32>, tensor<?x?xi32>, tensor<?x?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_indices
func @scatter_static_indices(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<3x6xi32>, %arg2 : tensor<?x?x?xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>) -> tensor<3x?x?xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<3x6xi32>, tensor<?x?x?xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_static_input
func @scatter_static_input(%arg0 : tensor<?x?x?xi32>, %arg1 : tensor<?x?xi32>, %arg2 : tensor<3x6x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>) -> tensor<3x?x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x?x?xi32>, tensor<?x?xi32>, tensor<3x6x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @scatter_minimum_static
func @scatter_minimum_static(%arg0 : tensor<?x4x?xi32>, %arg1 : tensor<3x?xi32>, %arg2 : tensor<?x?x5xi32>) {
  // CHECK: "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>) -> tensor<3x4x5xi32>
  %0 = "tosa.scatter"(%arg0, %arg1, %arg2) : (tensor<?x4x?xi32>, tensor<3x?xi32>, tensor<?x?x5xi32>)  -> (tensor<?x?x?xi32>)
  return
}

// -----

// CHECK-LABEL: @test_pool_static
func @test_pool_static(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x2x4x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x2x4x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_dynamic_input
func @test_pool_dynamic_input(%arg0: tensor<?x?x?x?xf32>) {
  // CHECK: -> tensor<?x?x?x?xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<?x?x?x?xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_padded
func @test_pool_padded(%arg0: tensor<3x5x6x7xf32>) {
  // CHECK: -> tensor<3x5x11x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [1, 2, 3, 4], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x5x11x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [1, 2, 3, 4], stride = [1, 1]} : (tensor<3x5x6x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

// -----

// CHECK-LABEL: @test_pool_stride
func @test_pool_stride(%arg0: tensor<3x11x12x7xf32>) {
  // CHECK: -> tensor<3x4x4x7xf32>
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [2, 3]} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>

  // CHECK: -> tensor<3x4x4x7xf32>
  %1 = "tosa.max_pool2d"(%arg0) {kernel = [4, 3], pad = [0, 0, 0, 0], stride = [2, 3]} : (tensor<3x11x12x7xf32>) -> tensor<?x?x?x?xf32>
  return
}

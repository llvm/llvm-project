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

func @test_dynamic_reshape(%arg0 : tensor<4x?xi32>) -> () {
  // CHECK: %0 = "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x?xi32>) -> tensor<16xi32>
  %0 = "tosa.reshape"(%arg0) {new_shape = [16]} : (tensor<4x?xi32>)  -> tensor<?xi32>

  // CHECK: %1 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x?xi32>) -> tensor<?xi32>
  %1 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<4x?xi32>)  -> tensor<?xi32>

  // CHECK: %2 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x?xi32>) -> tensor<2x?xi32>
  %2 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<4x?xi32>)  -> tensor<?x?xi32>

  return
}


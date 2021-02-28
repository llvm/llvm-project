// RUN: mlir-opt --split-input-file --tosa-to-linalg-on-tensors %s -verify-diagnostics -o -| FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>

// CHECK-LABEL: @test_abs
func @test_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [] : tensor<f32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = []} ins(%arg0 : tensor<f32>) outs([[INIT]] : tensor<f32>) {
  // CHECK: ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:   [[ELEMENT:%.+]] = absf %arg1
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<f32>

  %0 = "tosa.abs"(%arg0) : (tensor<f32>) -> tensor<f32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<f32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_abs
func @test_abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2] : tensor<2xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xf32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:   [[ELEMENT:%.+]] = absf %arg1
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_abs
func @test_abs(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x3xf32>) outs([[INIT]] : tensor<2x3xf32>) {
  // CHECK: ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:   [[ELEMENT:%.+]] = absf %arg1
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_broadcast
func @test_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2] : tensor<2xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1xf32>, tensor<2xf32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  // CHECK:   [[ELEMENT:%.+]] = addf %arg2, %arg3 : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_multibroadcast
func @test_multibroadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x3xf32>, tensor<2x1xf32>) outs([[INIT]] : tensor<2x3xf32>) {
  // CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  // CHECK:   [[ELEMENT:%.+]] = addf %arg2, %arg3 : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

func @test_abs(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{failed to legalize operation 'tosa.abs'}}
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @test_simple_f32
func @test_simple_f32(%arg0: tensor<1xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = "tosa.tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: absf
  %1 = "tosa.abs"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: addf
  %2 = "tosa.add"(%0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: subf
  %3 = "tosa.sub"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: mulf
  %4 = "tosa.mul"(%0, %1) {shift = 0 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: negf
  %5 = "tosa.negate"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: pow
  %6 = "tosa.pow"(%1, %2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: rsqrt
  %7 = "tosa.rsqrt"(%1) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: log
  %8 = "tosa.log"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: exp
  %9 = "tosa.exp"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: cmpf
  %10 = "tosa.greater"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: cmpf
  %11 = "tosa.greater_equal"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %12 = "tosa.select"(%10, %0, %1) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: cmpf
  // CHECK: select
  %13 = "tosa.maximum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: cmpf
  // CHECK: select
  %14 = "tosa.minimum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: ceil
  %15 = "tosa.ceil"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: floor
  %16 = "tosa.floor"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: cmpf
  // CHECK: select
  %17 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: cmpf
  // CHECK: select
  %18 = "tosa.reluN"(%0) {max_int = 5 : i64, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_i32
func @test_simple_i32(%arg0: tensor<1xi32>) -> () {
  // CHECK: linalg.generic
  // CHECK: addi
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: subi
  %1 = "tosa.sub"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: muli
  %2 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: muli
  %3 = "tosa.negate"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: and
  %4 = "tosa.bitwise_and"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: or
  %5 = "tosa.bitwise_or"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: xor
  %6 = "tosa.bitwise_xor"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: shift_left
  %7 = "tosa.logical_left_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: shift_right_unsigned
  %8 = "tosa.logical_right_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  %9 = "tosa.greater"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: cmpi
  %10 = "tosa.greater_equal"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %11 = "tosa.select"(%9, %0, %1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %12 = "tosa.maximum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %13 = "tosa.minimum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %14 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %15 = "tosa.reluN"(%0) {max_int = 5 : i64, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @test_reshape_downrank
func @test_reshape_downrank(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6]} : (tensor<2x3xf32>) -> tensor<6xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<6xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @test_reshape_uprank
func @test_reshape_uprank(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @test_reshape_samerank
func @test_reshape_samerank(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE1:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]]]
  // CHECK: [[RESHAPE2:%.+]] = linalg.tensor_reshape [[RESHAPE1]] [#[[$MAP0]]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE2]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>

// CHECK-LABEL: @test_reshape_downrank_6D
func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  // CHECK: linalg.tensor_reshape %arg0 [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_identity
func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0 = "tosa.identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tosa.identity"(%arg1) : (tensor<1xi32>) -> tensor<1xi32>

  %2:2 = "tosa.identityn"(%0, %1) : (tensor<1xf32>, tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>)

  // CHECK: return %arg0, %arg1
  return %2#0, %2#1 : tensor<1xf32>, tensor<1xi32>
}

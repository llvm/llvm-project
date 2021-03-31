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

  // CHECK: linalg.generic
  // CHECK: negf
  // CHECK: exp
  // CHECK: addf
  // CHECK: divf
  %19 = "tosa.sigmoid"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: fptosi
  %20 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: constant 0
  // CHECK: cmpf
  %21 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: fptrunc
  %22 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xf16>

  // CHECK: linalg.generic
  // CHECK: yield
  %23 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: divf
  %24 = "tosa.reciprocal"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_f16
func @test_simple_f16(%arg0: tensor<1xf16>) -> () {

  // CHECK: linalg.generic
  // CHECK: fpext
  %0 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_i16
func @test_simple_i16(%arg0: tensor<1xi16>) -> () {
  // CHECK: linalg.generic
  // CHECK: sext
  // CHECK: sext
  // CHECK: muli
  %0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi32>

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
  // CHECK: constant 2
  // CHECK: apply_scale
  %3 = "tosa.mul"(%arg0, %arg0) {shift = 2 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: muli
  %4 = "tosa.negate"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: and
  %5 = "tosa.bitwise_and"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: or
  %6 = "tosa.bitwise_or"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: xor
  %7 = "tosa.bitwise_xor"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: shift_left
  %8 = "tosa.logical_left_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: shift_right_unsigned
  %9 = "tosa.logical_right_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  %10 = "tosa.greater"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: cmpi
  %11 = "tosa.greater_equal"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %12 = "tosa.select"(%10, %0, %1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %13 = "tosa.maximum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %14 = "tosa.minimum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %15 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: cmpi
  // CHECK: select
  %16 = "tosa.reluN"(%0) {max_int = 5 : i64, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: trunci
  %17 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi16>

  // CHECK: linalg.generic
  // CHECK: yield
  %18 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: sexti
  %19 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi64>

  // CHECK: linalg.generic
  // CHECK: constant 0
  // CHECK: cmpi
  %20 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: sitofp
  %21 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_bool
func @test_bool(%arg0: tensor<1xi1>, %arg1: tensor<1xi1>) -> () {
  // CHECK: linalg.generic
  // CHECK: and
  %0 = "tosa.logical_and"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: or
  %1 = "tosa.logical_or"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: xor
  %2 = "tosa.logical_xor"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: constant true
  // CHECK: xor
  %3 = "tosa.logical_not"(%arg0) : (tensor<1xi1>) -> tensor<1xi1>

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

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @test_transpose
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2x3xi32>)
func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
  %0 = constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 3, 1]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x2x3xi32>) outs([[OUT:%.+]] : tensor<2x3x1xi32>)
  // CHECK: ^bb0([[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32)
  // CHECK:   linalg.yield [[ARG1]]
  // CHECK: }
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x3xi32>, tensor<3xi32>) -> (tensor<2x3x1xi32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL: @reduce_float
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xf32>
func @reduce_float(%arg0: tensor<5x4xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 4]
  // CHECK: [[CST0:%.+]] = constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[INIT]], [[CST0]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<1x4xf32>)
  // CHECK: ^bb0(%arg1: f32, %arg2: f32)
  // CHECK:   [[RES:%.+]] = addf %arg1, %arg2 : f32
  // CHECK:   linalg.yield [[RES]] : f32
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 1]
  // CHECK: [[CST0:%.+]] = constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[INIT]], [[CST0]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<5x1xf32>)
  // CHECK: ^bb0(%arg1: f32, %arg2: f32)
  // CHECK:   [[RES:%.+]] = addf %arg1, %arg2 : f32
  // CHECK:   linalg.yield [[RES]] : f32
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<5x4xf32>) -> tensor<5x1xf32>

  // CHECK: constant 1.0
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: mulf
  %2 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: constant 3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: cmpf olt
  // CHECK: select
  %3 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: constant -3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: cmpf ogt
  // CHECK: select
  %4 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK-LABEL: @reduce_int
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi32>
func @reduce_int(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 4]
  // CHECK: [[CST0:%.+]] = constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[INIT]], [[CST0]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<1x4xi32>)
  // CHECK: ^bb0(%arg1: i32, %arg2: i32)
  // CHECK:   [[RES:%.+]] = addi %arg1, %arg2 : i32
  // CHECK:   linalg.yield [[RES]] : i32
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 1]
  // CHECK: [[CST0:%.+]] = constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[INIT]], [[CST0]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<5x1xi32>)
  // CHECK: ^bb0(%arg1: i32, %arg2: i32)
  // CHECK:   [[RES:%.+]] = addi %arg1, %arg2 : i32
  // CHECK:   linalg.yield [[RES]] : i32
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<5x4xi32>) -> tensor<5x1xi32>

  // CHECK: constant 1
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: muli
  %2 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: constant 2147483647 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: cmpi slt
  // CHECK: select
  %3 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: constant -2147483648 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: cmpi sgt
  // CHECK: select
  %4 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, d1)>

// CHECK-LABEL: @reduce_bool
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi1>
func @reduce_bool(%arg0: tensor<5x4xi1>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 4]
  // CHECK: [[CST0:%.+]] = constant true
  // CHECK: [[FILL:%.+]] = linalg.fill([[INIT]], [[CST0]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi1>) outs([[FILL]] : tensor<1x4xi1>)
  // CHECK: ^bb0(%arg1: i1, %arg2: i1)
  // CHECK:   [[RES:%.+]] = and %arg1, %arg2 : i1
  // CHECK:   linalg.yield [[RES]] : i1
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  // CHECK: constant false
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: or
  %1 = "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  return
}

// -----

// CHECK-LABEL: @concat
func @concat(%arg0: tensor<5x1xf32>, %arg1: tensor<6x1xf32>) -> () {
  // CHECK: [[AXIS:%.+]] = constant 0
  // CHECK: [[STRIDE:%.+]]   = constant 1
  // CHECK: [[OFFSET:%.+]] = constant 0 : index
  // CHECK: [[IDX0:%.+]] = constant 0 : index
  // CHECK: [[ARG0_DIM0:%.+]] = memref.dim %arg0, [[IDX0]]
  // CHECK: [[IDX1:%.+]] = constant 1 : index
  // CHECK: [[ARG0_DIM1:%.+]] = memref.dim %arg0, [[IDX1]]
  // CHECK: [[ARG1_AXIS:%.+]] = memref.dim %arg1, [[AXIS]]
  // CHECK: [[RESULT_AXIS:%.+]] = addi [[ARG0_DIM0]], [[ARG1_AXIS]]
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [11, 1]
  // CHECK: [[ARG0_DIM0:%.+]] = memref.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT0:%.+]] = subtensor_insert %arg0 into [[INIT]]{{\[}}[[OFFSET]], [[OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  // CHECK: [[NEW_OFFSET:%.+]] = addi [[OFFSET]], [[ARG0_DIM0]]
  // CHECK: [[ARG1_DIM0:%.+]] = memref.dim %arg1, [[AXIS]]
  // CHECK: [[INSERT1:%.+]] = subtensor_insert %arg1 into [[INSERT0]]{{\[}}[[NEW_OFFSET]], [[OFFSET]]] {{\[}}[[ARG1_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<5x1xf32>, tensor<6x1xf32>)  -> (tensor<11x1xf32>)

  // CHECK: [[AXIS:%.+]] = constant 1
  // CHECK: [[STRIDE:%.+]]   = constant 1
  // CHECK: [[OFFSET:%.+]] = constant 0 : index
  // CHECK: [[IDX0:%.+]] = constant 0 : index
  // CHECK: [[ARG0_DIM0:%.+]] = memref.dim %arg0, [[IDX0]]
  // CHECK: [[IDX1:%.+]] = constant 1 : index
  // CHECK: [[ARG0_DIM1:%.+]] = memref.dim %arg0, [[IDX1]]
  // CHECK: [[ARG1_AXIS:%.+]] = memref.dim %arg0, [[AXIS]]
  // CHECK: [[RESULT_AXIS:%.+]] = addi [[ARG0_DIM1]], [[ARG1_AXIS]]
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 2]
  // CHECK: [[ARG0_DIM1:%.+]] = memref.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT0:%.+]] = subtensor_insert %arg0 into [[INIT]]{{\[}}[[OFFSET]], [[OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  // CHECK: [[NEW_OFFSET:%.+]] = addi [[OFFSET]], [[ARG0_DIM1]]
  // CHECK: [[ARG1_DIM1:%.+]] = memref.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT1:%.+]] = subtensor_insert %arg0 into [[INSERT0]]{{\[}}[[OFFSET]], [[NEW_OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG1_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  %1 = "tosa.concat"(%arg0, %arg0) { axis = 1 : i64} : (tensor<5x1xf32>, tensor<5x1xf32>)  -> (tensor<5x2xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK-LABEL: @rescale
func @rescale(%arg0 : tensor<1xi8>) -> (tensor<1xi8>) {
  // CHECK: [[C0:%.+]] = constant dense<19689>
  // CHECK: [[C1:%.+]] = constant dense<15>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0, [[C0]], [[C1]] : tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) outs([[INIT]] : tensor<1xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[MULTIPLIER:%.+]]: i32, [[SHIFT:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C243:%.+]] = constant 243
  // CHECK: [[C252:%.+]] = constant 252

  // CHECK-DAG: [[IN32:%.+]] = sexti [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = subi [[IN32]], [[C243]]
  // CHECK-DAG: [[SCALED:%.+]] = "tosa.apply_scale"([[IN_ZEROED]], [[MULTIPLIER]], [[SHIFT]]) {double_round = false}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = addi [[SCALED]], [[C252]]
  // CHECK-DAG: [[CMIN:%.+]] = constant -128
  // CHECK-DAG: [[CMAX:%.+]] = constant 127
  // CHECK-DAG: [[MINLT:%.+]] = cmpi slt, [[SCALED_ZEROED]], [[CMIN]]
  // CHECK-DAG: [[MAXLT:%.+]] = cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[LOWER:%.+]] = select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<1xi8>)  -> (tensor<1xi8>)

  // CHECK: return [[GENERIC]]
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: @rescaleDoubleRound
func @rescaleDoubleRound(%arg0 : tensor<1xi8>) -> (tensor<1xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = true}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [33 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<1xi8>)  -> (tensor<1xi8>)
  return %0 : tensor<1xi8>
}

// CHECK-LABEL: @rescaleUnnecessaryDoubleRound
func @rescaleUnnecessaryDoubleRound(%arg0 : tensor<1xi8>) -> (tensor<1xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = false}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<1xi8>)  -> (tensor<1xi8>)
  return %0 : tensor<1xi8>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (-d0 + 4, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 3)>

// CHECK-LABEL: @reverse
func @reverse(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 4]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x4xi32>) outs([[INIT]] : tensor<5x4xi32>) {
  // CHECK: ^bb0(%arg1: i32, %arg2: i32):
  // CHECK:   linalg.yield %arg1 : i32
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 4]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x4xi32>) outs([[INIT]] : tensor<5x4xi32>) {
  // CHECK: ^bb0(%arg1: i32, %arg2: i32):
  // CHECK:   linalg.yield %arg1 : i32
  %1 = "tosa.reverse"(%arg0) {axis = 1 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, 0, d3)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP4:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

// CHECK-LABEL: @tile
func @tile(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]], #[[$MAP1]]] : tensor<2x3xi8> into tensor<1x2x1x3xi8>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 2, 1, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RESHAPE]] : tensor<1x2x1x3xi8>) outs([[INIT]] : tensor<2x2x1x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_reshape [[GENERIC]] [#[[$MAP0]], #[[$MAP1]]]
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1]} : (tensor<2x3xi8>)  -> (tensor<4x3xi8>)

  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]], #[[$MAP1]]] : tensor<2x3xi8> into tensor<1x2x1x3xi8>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 2, 2, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RESHAPE]] : tensor<1x2x1x3xi8>) outs([[INIT]] : tensor<1x2x2x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_reshape [[GENERIC]] [#[[$MAP4]], #[[$MAP5]]]
  %1 = "tosa.tile"(%arg0) {multiples = [1, 2]} : (tensor<2x3xi8>)  -> (tensor<2x6xi8>)

  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_reshape %arg0 [#[[$MAP0]], #[[$MAP1]]] : tensor<2x3xi8> into tensor<1x2x1x3xi8>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 2, 7, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[RESHAPE]] : tensor<1x2x1x3xi8>) outs([[INIT]] : tensor<5x2x7x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_reshape [[GENERIC]] [#[[$MAP4]], #[[$MAP5]]]
  %2 = "tosa.tile"(%arg0) {multiples = [5, 7]} : (tensor<2x3xi8>)  -> (tensor<10x21xi8>)

  return
}

// -----


// CHECK-LABEL: @matmul
func @matmul(%arg0: tensor<5x3xf32>, %arg1: tensor<3x6xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  // CHECK: [[C0:%.+]] = constant 0
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[FILLED:%.+]] = linalg.fill([[INIT]], [[C0]]) : tensor<5x6xf32>, f32 -> tensor<5x6xf32>
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<5x3xf32>, tensor<3x6xf32>) outs([[FILLED]] : tensor<5x6xf32>) -> tensor<5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<5x3xf32>, tensor<3x6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, d1)>

// CHECK-LABEL: @fully_connected
func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<3x6xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  // CHECK: [[RS:%.+]] = linalg.tensor_reshape %arg2 [#[[$MAP0]]]
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins([[RS]] : tensor<1x6xf32>) outs([[INIT]] : tensor<5x6xf32>) {
  // CHECK: ^bb0([[IN:%.+]]: f32, [[MULTIPLIER:%.+]]: f32):
  // CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<5x3xf32>, tensor<3x6xf32>) outs([[GENERIC]] : tensor<5x6xf32>) -> tensor<5x6xf32>
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<5x3xf32>, tensor<3x6xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}

// -----

func @pad_float(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[INDEX0:%.+]] = constant 0 : index
  // CHECK: [[INDEX1:%.+]] = constant 1 : index
  // CHECK: [[ROW0:%.+]] = constant 0 : index
  // CHECK: [[LOW0:%.+]] = tensor.extract %cst{{\[}}[[ROW0]], [[INDEX0]]]
  // CHECK: [[HIGH0:%.+]] = tensor.extract %cst{{\[}}[[ROW0]], [[INDEX1]]]
  // CHECK: [[LOW0_IDX:%.+]] = index_cast %0
  // CHECK: [[HIGH0_IDX:%.+]] = index_cast %1
  // CHECK: [[ROW1:%.+]] = constant 1 : index
  // CHECK: [[LOW1:%.+]] = tensor.extract %cst{{\[}}%c1_1, %c0]
  // CHECK: [[HIGH1:%.+]] = tensor.extract %cst{{\[}}%c1_1, %c1]
  // CHECK: [[LOW1_IDX:%.+]] = index_cast [[LOW1]]
  // CHECK: [[HIGH1_IDX:%.+]] = index_cast [[HIGH1]]
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: %8 = linalg.pad_tensor %arg0 low{{\[}}[[LOW0_IDX]], [[LOW1_IDX]]] high{{\[}}[[HIGH0_IDX]], [[HIGH1_IDX]]]  {
  // CHECK: ^bb0(%arg1: index, %arg2: index):  // no predecessors
  // CHECK:   linalg.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<4x9xf32>)
  return %1 : tensor<4x9xf32>
}

func @pad_int(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = constant 0 : i32
  // CHECK: linalg.pad_tensor
  // CHECK:   linalg.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

func @pad_quant(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = constant 42 : i32
  // CHECK: linalg.pad_tensor
  // CHECK:   linalg.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0) { quantization_info = { input_zp = 42 : i32}} : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP4:.*]] = affine_map<(d0) -> ()>

func @argmax(%arg0 : tensor<3x2xi32>, %arg1 : tensor<6xf32>) -> () {
  // CHECK: [[IDX_INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[IDX_MIN:%.+]] = constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill([[IDX_INIT]], [[IDX_MIN]])
  // CHECK: [[VAL_INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[VAL_MIN:%.+]] = constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill([[VAL_INIT]], [[VAL_MIN]])
  // CHECK: linalg.indexed_generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<2xi32>, tensor<2xi32>)
  // CHECK:   [[CAST:%.+]] = index_cast %arg2
  // CHECK:   [[CMP:%.+]] = cmpi sgt, %arg4, %arg6
  // CHECK:   [[SELECT_VAL:%.+]] = select [[CMP]], %arg4, %arg6
  // CHECK:   [[SELECT_IDX:%.+]] = select [[CMP]], [[CAST]], %arg5
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %0 = "tosa.argmax"(%arg0) { axis = 0 : i64} : (tensor<3x2xi32>)  -> (tensor<2xi32>)

  // CHECK: [[IDX_INIT:%.+]] = linalg.init_tensor [3]
  // CHECK: [[IDX_MIN:%.+]] = constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill([[IDX_INIT]], [[IDX_MIN]])
  // CHECK: [[VAL_INIT:%.+]] = linalg.init_tensor [3]
  // CHECK: [[VAL_MIN:%.+]] = constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill([[VAL_INIT]], [[VAL_MIN]])
  // CHECK: linalg.indexed_generic {indexing_maps = [#map0, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK:   [[CAST:%.+]] = index_cast %arg3
  // CHECK:   [[CMP:%.+]] = cmpi sgt, %arg4, %arg6
  // CHECK:   [[SELECT_VAL:%.+]] = select [[CMP]], %arg4, %arg6
  // CHECK:   [[SELECT_IDX:%.+]] = select [[CMP]], [[CAST]], %arg5
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %1 = "tosa.argmax"(%arg0) { axis = 1 : i64} : (tensor<3x2xi32>)  -> (tensor<3xi32>)

  // CHECK: constant -3.40282347E+38 : f32
  // CHECK: index_cast
  // CHECK: cmpf ogt
  // CHECK: select
  // CHECK: select
  // CHECK: linalg.yield
  %2 = "tosa.argmax"(%arg1) { axis = 0 : i64} : (tensor<6xf32>)  -> (tensor<i32>)

  return
}

// -----

// CHECK-LABEL: @table8
func @table8(%arg0: tensor<6xi8>, %arg1: tensor<513xi8>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = index_cast %[[ARG_IN]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %arg1[%[[CAST]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi8>, tensor<513xi8>)  -> (tensor<6xi8>)
  return
}

// CHECK-LABEL: @table16
func @table16(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<6xi16>) outs(%[[INIT]] : tensor<6xi32>)
  // CHECK: ^bb0(%arg2: i16, %arg3: i32)
  // CHECK: %[[EXT_IN:.+]] = sexti %arg2
  // CHECK: %[[C32768:.+]] = constant 32768
  // CHECK: %[[C7:.+]] = constant 7
  // CHECK: %[[C1:.+]] = constant 1
  // CHECK: %[[C127:.+]] = constant 127
  // CHECK: %[[INADD:.+]] = addi %[[EXT_IN]], %[[C32768]]
  // CHECK: %[[IDX:.+]] = shift_right_unsigned %[[INADD]], %[[C7]]
  // CHECK: %[[FRACTION:.+]] = and %[[INADD]], %[[C127]]
  // CHECK: %[[IDXPLUS1:.+]] = addi %[[IDX]], %[[C1]]
  // CHECK: %[[IDX_CAST:.+]] = index_cast %[[IDX]]
  // CHECK: %[[IDXPLUS1_CAST:.+]] = index_cast %[[IDXPLUS1]]
  // CHECK: %[[BASE:.+]] = tensor.extract %arg1[%[[IDX_CAST]]]
  // CHECK: %[[NEXT:.+]] = tensor.extract %arg1[%[[IDXPLUS1_CAST]]]
  // CHECK: %[[BASE_EXT:.+]] = sexti %[[BASE]]
  // CHECK: %[[NEXT_EXT:.+]] = sexti %[[NEXT]]
  // CHECK: %[[BASE_MUL:.+]] = shift_left %[[BASE_EXT]], %[[C7]]
  // CHECK: %[[DIFF:.+]] = subi %[[NEXT_EXT]], %[[BASE_EXT]]
  // CHECK: %[[DIFF_MUL:.+]] = muli %[[DIFF]], %[[FRACTION]]
  // CHECK: %[[RESULT:.+]] = addi %[[BASE_MUL]], %[[DIFF_MUL]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi16>, tensor<513xi16>)  -> (tensor<6xi32>)
  return
}

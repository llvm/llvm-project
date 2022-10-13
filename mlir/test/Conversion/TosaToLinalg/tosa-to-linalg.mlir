// RUN: mlir-opt --split-input-file -pass-pipeline="func.func(tosa-to-linalg)" %s -verify-diagnostics -o -| FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>

// CHECK-LABEL: @test_abs
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<f32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = []} ins(%[[ARG0]] : tensor<f32>) outs([[INIT]] : tensor<f32>) {
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = math.absf %[[ARG1]]
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<f32>

  %0 = "tosa.abs"(%arg0) : (tensor<f32>) -> tensor<f32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<f32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_abs
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xf32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = math.absf %[[ARG1]]
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_abs
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xf32>) outs([[INIT]] : tensor<2x3xf32>) {
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = math.absf %[[ARG1]]
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_abs
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM]])
  // CHECK: linalg.generic
  // CHECK: math.absf
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_abs_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_abs_dyn(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM]])
  // CHECK: linalg.generic
  // CHECK: math.absf
  %0 = "tosa.abs"(%arg0) : (tensor<2x?xf32>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}
// -----


// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> ()>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_broadcast
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<1xf32
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]: tensor<2xf32>
func.func @test_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2xf32>
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG0]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins([[RESHAPE]], %[[ARG1]] : tensor<f32>, tensor<2xf32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> ()>

// CHECK-LABEL: @test_broadcast_swapped_args
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<2xf32
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]: tensor<1xf32>
func.func @test_broadcast_swapped_args(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2xf32>
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG1]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], [[RESHAPE]] : tensor<2xf32>, tensor<f32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<1xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @test_multibroadcast
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @test_multibroadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: [[RESHAPE1:%.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK: [[RESHAPE2:%.+]] = tensor.collapse_shape %[[ARG1]] {{\[}}[0, 1]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins([[RESHAPE1]], [[RESHAPE2]] : tensor<3xf32>, tensor<2xf32>) outs([[INIT]] : tensor<2x3xf32>) {
  // CHECK: ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %[[ARG2]], %[[ARG3]] : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_simple_f32
func.func @test_simple_f32(%arg0: tensor<1xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = "tosa.tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.absf
  %1 = "tosa.abs"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.addf
  %2 = "tosa.add"(%0, %0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.subf
  %3 = "tosa.sub"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %4 = "tosa.mul"(%0, %1) {shift = 0 : i32} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.negf
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
  // CHECK: arith.cmpf
  %10 = "tosa.greater"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  %11 = "tosa.greater_equal"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  %12 = "tosa.equal"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %13 = "tosa.select"(%10, %0, %1) : (tensor<1xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.maxf
  %14 = "tosa.maximum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.minf
  %15 = "tosa.minimum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: ceil
  %16 = "tosa.ceil"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: floor
  %17 = "tosa.floor"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.minf
  // CHECK: arith.maxf
  %18 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: exp
  // CHECK: arith.addf
  // CHECK: arith.divf
  %19 = "tosa.sigmoid"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0.000000e+00
  // CHECK: arith.constant 5.000000e-01
  // CHECK: arith.constant -2.14748365E+9
  // CHECK: arith.constant 2.14748365E+9
  // CHECK: arith.addf
  // CHECK: arith.subf
  // CHECK: arith.cmpf olt
  // CHECK: select
  // CHECK: arith.minf
  // CHECK: arith.maxf
  // CHECK: arith.fptosi
  %20 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpf
  %21 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.truncf
  %22 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xf16>

  // CHECK: linalg.generic
  // CHECK: arith.divf
  %23 = "tosa.reciprocal"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_f16
func.func @test_simple_f16(%arg0: tensor<1xf16>) -> () {

  // CHECK: linalg.generic
  // CHECK: arith.extf
  %0 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_i16
func.func @test_simple_i16(%arg0: tensor<1xi16>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.extsi
  // CHECK: arith.extsi
  // CHECK: arith.muli
  %0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func.func @test_simple_ui8(%arg0: tensor<1xui8>) -> () {
  // CHECK: arith.uitofp
  %0 = "tosa.cast"(%arg0) : (tensor<1xui8>) -> tensor<1xf32>
  return
}

// -----

// CHECK-LABEL: @test_simple_i32
func.func @test_simple_i32(%arg0: tensor<1xi32>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.addi
  %0 = "tosa.add"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.subi
  %1 = "tosa.sub"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.muli
  %2 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 2
  // CHECK: apply_scale
  %3 = "tosa.mul"(%arg0, %arg0) {shift = 2 : i32} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.divsi
  %4 = "tosa.div"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32):
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: arith.subi [[ZERO]], %[[ARG1]]
  %5 = "tosa.negate"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: and
  %6 = "tosa.bitwise_and"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: or
  %7 = "tosa.bitwise_or"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.xori
  %8 = "tosa.bitwise_xor"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shli
  %9 = "tosa.logical_left_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shrui
  %10 = "tosa.logical_right_shift"(%arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.shrsi
  %11 = "tosa.arithmetic_right_shift"(%arg0, %arg0) {round = 0 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 1
  // CHECK: arith.constant 0
  // CHECK: arith.constant true
  // CHECK: arith.cmpi
  // CHECK: arith.subi
  // CHECK: arith.shrsi
  // CHECK: arith.trunci
  // CHECK: and
  // CHECK: and
  // CHECK: arith.extui
  // CHECK: arith.addi
  %12 = "tosa.arithmetic_right_shift"(%arg0, %arg0) {round = 1 : i1} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: math.ctlz
  %13 = "tosa.clz"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  %14 = "tosa.greater"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  %15 = "tosa.greater_equal"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: select
  %16 = "tosa.select"(%14, %0, %1) : (tensor<1xi1>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  // CHECK: select
  %17 = "tosa.maximum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  // CHECK: select
  %18 = "tosa.minimum"(%0, %1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpi
  // CHECK: select
  %19 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant -32768
  // CHECK: arith.constant 32767
  // CHECK: arith.cmpi slt
  // CHECK: select
  // CHECK: arith.cmpi slt
  // CHECK: select
  // CHECK: arith.trunci
  %20 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi16>

  // CHECK: linalg.generic
  // CHECK: arith.extsi
  %21 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi64>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpi
  %22 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.sitofp
  %23 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpi sgt
  // CHECK: arith.subi
  // CHECK: select
  %24 = "tosa.abs"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func.func @test_simple_ui8(%arg0: tensor<1xi8>) -> () {

  // CHECK: linalg.generic
  // CHECK: sitofp
  %0 = "tosa.cast"(%arg0) : (tensor<1xi8>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_i8
func.func @test_i8(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK-DAG: %[[C127:.+]] = arith.constant -127
  // CHECK-DAG: %[[C126:.+]] = arith.constant 126
  // CHECK-DAG: %[[CMP1:.+]] = arith.cmpi slt, %[[ARG1]], %[[C127]]
  // CHECK-DAG: %[[SEL1:.+]] = arith.select %[[CMP1]], %[[C127]]
  // CHECK-DAG: %[[CMP2:.+]] = arith.cmpi slt, %[[C126]], %[[ARG1]]
  // CHECK: %[[SEL2:.+]] = arith.select %[[CMP2]], %[[C126]], %[[SEL1]]
  %0 = "tosa.clamp"(%arg0) {min_int = -127 : i64, max_int = 126 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK-DAG: %[[C128:.+]] = arith.constant -128
  // CHECK-DAG: %[[C127:.+]] = arith.constant 127
  // CHECK-DAG: %[[CMP1:.+]] = arith.cmpi slt, %[[ARG1]], %[[C128]]
  // CHECK-DAG: %[[SEL1:.+]] = arith.select %[[CMP1]], %[[C128]]
  // CHECK-DAG: %[[CMP2:.+]] = arith.cmpi slt, %[[C127]], %[[ARG1]]
  // CHECK: %[[SEL2:.+]] = arith.select %[[CMP2]], %[[C127]], %[[SEL1]]
  %1 = "tosa.clamp"(%arg0) {min_int = -130 : i64, max_int = 130 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>

  return
}

// -----

// CHECK-LABEL: @test_clamp_f16
func.func @test_clamp_f16(%arg0: tensor<1xf16>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG1:.+]]: f16,
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0.0
  // CHECK-DAG: %[[C6:.+]] = arith.constant 6.0
  // CHECK-DAG: %[[MIN:.+]] = arith.minf %[[ARG1]], %[[C6]]
  // CHECK-DAG: %[[MAX:.+]] = arith.maxf %[[MIN]], %[[C0]]
  %0 = "tosa.clamp"(%arg0) {min_int = 0 : i64, max_int = 0 : i64, min_fp = 0.0 : f32, max_fp = 6.0 : f32} : (tensor<1xf16>) -> tensor<1xf16>

  return
}

// -----

// CHECK-LABEL: @test_bool
func.func @test_bool(%arg0: tensor<1xi1>, %arg1: tensor<1xi1>) -> () {
  // CHECK: linalg.generic
  // CHECK: and
  %0 = "tosa.logical_and"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: or
  %1 = "tosa.logical_or"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.xori
  %2 = "tosa.logical_xor"(%arg0, %arg1) : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.constant true
  // CHECK: arith.xori
  %3 = "tosa.logical_not"(%arg0) : (tensor<1xi1>) -> tensor<1xi1>

  return
}

// -----

// CHECK-LABEL: @test_negate_quantized
func.func @test_negate_quantized(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8,
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[EXT:%.+]] = arith.extsi %[[BBARG0]] : i8 to i16
  // CHECK: [[SUB:%.+]] = arith.subi [[ZERO]], [[EXT]]
  // CHECK: [[MIN:%.+]] = arith.constant -128
  // CHECK: [[MAX:%.+]] = arith.constant 127
  // CHECK: [[PRED1:%.+]] = arith.cmpi slt, [[SUB]], [[MIN]]
  // CHECK: [[LBOUND:%.+]] = arith.select [[PRED1]], [[MIN]], [[SUB]]
  // CHECK: [[PRED2:%.+]] = arith.cmpi slt, [[MAX]], [[SUB]]
  // CHECK: [[UBOUND:%.+]] = arith.select [[PRED2]], [[MAX]], [[LBOUND]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[UBOUND]]
  // CHECK: linalg.yield [[TRUNC]]
  %0 = "tosa.negate"(%arg0) {quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8,
  // CHECK: [[EXT:%.+]] = arith.extsi %[[BBARG0]] : i8 to i16
  %1 = "tosa.negate"(%arg0) {quantization_info = #tosa.unary_quant<input_zp = 32639, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[BBARG0:.+]]: i8,
  // CHECK: [[EXT:%.+]] = arith.extsi %[[BBARG0]] : i8 to i32
  %2 = "tosa.negate"(%arg0) {quantization_info = #tosa.unary_quant<input_zp = 32640, output_zp = 0>} : (tensor<1xi8>) -> tensor<1xi8>

  return
}

// -----

// CHECK-LABEL: @test_reshape_downrank
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_downrank(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6]} : (tensor<2x3xf32>) -> tensor<6xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_downrank_dyn(%arg0: tensor<2x?xf32>) -> tensor<?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<2x?xf32>) -> tensor<?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_uprank(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
func.func @test_reshape_uprank_dyn(%arg0: tensor<?xf32>) -> tensor<2x?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<?xf32>) -> tensor<2x?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xf32>)
func.func @test_reshape_samerank(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank_dyn
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<?x2xf32>)
func.func @test_reshape_samerank_dyn(%arg0: tensor<?x2xf32>) -> tensor<2x?xf32> {
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<?x2xf32>) -> tensor<2x?xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  // CHECK: tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3], [4, 5]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6, 5, 77]} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @test_reshape_downrank_6D_dyn(%arg0: tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32> {
  // CHECK: tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2, 3, 4, 5]]
  // CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1, 5, 77]} : (tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32>
  return %0 : tensor<?x5x77xf32>
}

// -----

// CHECK-LABEL: @test_identity
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<1xf32>,
// CHECK-SAME: %[[ARG1:[0-9a-zA-Z_]*]]: tensor<1xi32>
func.func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0 = "tosa.identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tosa.identity"(%arg1) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: return %[[ARG0]], %[[ARG1]]
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @test_transpose
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2x3xi32>)
func.func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
  %0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<2x3x1xi32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[ARG0]] : tensor<1x2x3xi32>) outs([[OUT:%.+]] : tensor<2x3x1xi32>)
  // CHECK: ^bb0([[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32)
  // CHECK:   linalg.yield [[ARG1]]
  // CHECK: }
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x3xi32>, tensor<3xi32>) -> (tensor<2x3x1xi32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @test_transpose_dyn
// CHECK-SAME: (%[[ARG0:.+]]: tensor<1x?x3x4xi32>)
func.func @test_transpose_dyn(%arg0: tensor<1x?x3x4xi32>) -> () {
  %0 = arith.constant dense<[1, 3, 0, 2]> : tensor<4xi32>
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM]]) : tensor<?x4x1x3xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<1x?x3x4xi32>) outs([[OUT:%.+]] : tensor<?x4x1x3xi32>)
  // CHECK: ^bb0([[ARG1:%.+]]: i32, [[ARG2:%.+]]: i32)
  // CHECK:   linalg.yield [[ARG1]]
  // CHECK: }
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x?x3x4xi32>, tensor<4xi32>) -> (tensor<?x4x1x3xi32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_transpose_dyn
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>)
func.func @test_transpose_dyn_multiple(%arg0: tensor<?x?xf32>) -> () {
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM0]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x?xf32>) outs([[OUT:%.+]] : tensor<?x?xf32>)
  // CHECK: ^bb0([[ARG1:%.+]]: f32, [[ARG2:%.+]]: f32)
  // CHECK:   linalg.yield [[ARG1]]
  // CHECK: }
  %1 = "tosa.transpose"(%arg0, %0) : (tensor<?x?xf32>, tensor<2xi32>) -> (tensor<?x?xf32>)
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @reduce_float
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xf32>
func.func @reduce_float(%arg0: tensor<5x4xf32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<4xf32>
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<4xf32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   [[RES:%.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK: tensor.expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<5xf32>
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<5xf32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   [[RES:%.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK: tensor.expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<5xf32> into tensor<5x1xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<5x4xf32>) -> tensor<5x1xf32>

  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %2 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant 3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.minf
  %3 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant -3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.maxf
  %4 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: @reduce_float_dyn
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<?x5x4xf32>
func.func @reduce_float_dyn(%arg0: tensor<?x5x4xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<?x4xf32>
  // CHECK: %[[CST0:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[ARG0]] : tensor<?x5x4xf32>) outs(%[[FILL]] : tensor<?x4xf32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   %[[RES:.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK: tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1, 2]] : tensor<?x4xf32> into tensor<?x1x4xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<?x5x4xf32>) -> tensor<?x1x4xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> ()>

// CHECK-LABEL: @reduce_float_dyn_rank_1
// CHECK-SAME: %[[ARG0:[0-9a-zA-Z_]*]]: tensor<?xf32>
func.func @reduce_float_dyn_rank_1(%arg0: tensor<?xf32>) -> () {
  // CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<f32>
  // CHECK-DAG: %[[CST0:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[ARG0]] : tensor<?xf32>) outs(%[[FILL]] : tensor<f32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   %[[RES:.+]] = arith.addf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK: tensor.expand_shape %[[GENERIC]] {{\[}}] : tensor<f32> into tensor<1xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<?xf32>) -> tensor<1xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: @reduce_float_dyn_nonzero_batch
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reduce_float_dyn_nonzero_batch(%arg0: tensor<5x?x4xf32>) -> () {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<5x?xf32>
  // CHECK: %[[CST1:.+]] = arith.constant 1.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[ARG0]] : tensor<5x?x4xf32>) outs(%[[FILL]] : tensor<5x?xf32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   %[[RES:.+]] = arith.mulf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[RES]] : f32
  // CHECK: tensor.expand_shape %[[GENERIC]] {{\[}}[0], [1, 2]] : tensor<5x?xf32> into tensor<5x?x1xf32>
  %0 = "tosa.reduce_prod"(%arg0) {axis = 2 : i64} : (tensor<5x?x4xf32>) -> tensor<5x?x1xf32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @reduce_float_dyn_multiple
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @reduce_float_dyn_multiple(%arg0: tensor<?x?xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[CMIN:.+]] = arith.constant -3.40282347E+38
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CMIN]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[ARG0]] : tensor<?x?xf32>) outs(%[[FILL]] : tensor<?xf32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
  // CHECK:   %[[MAX:.+]] = arith.maxf %[[ARG1]], %[[ARG2]] : f32
  // CHECK:   linalg.yield %[[MAX]] : f32
  // CHECK: tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1]] : tensor<?xf32> into tensor<?x1xf32>
  %0 = "tosa.reduce_max"(%arg0) {axis = 1 : i64} : (tensor<?x?xf32>) -> tensor<?x1xf32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @reduce_int
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi32>
func.func @reduce_int(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<4xi32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
  // CHECK:   [[RES:%.+]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK: tensor.expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<5xi32>)
  // CHECK: ^bb0(%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32)
  // CHECK:   [[RES:%.+]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK: tensor.expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<5xi32> into tensor<5x1xi32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<5x4xi32>) -> tensor<5x1xi32>

  // CHECK: arith.constant 1
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.muli
  %2 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: arith.constant 2147483647 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.cmpi slt
  // CHECK: select
  %3 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: arith.constant -2147483648 : i32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.cmpi sgt
  // CHECK: select
  %4 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: @reduce_bool
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi1>
func.func @reduce_bool(%arg0: tensor<5x4xi1>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[CST0:%.+]] = arith.constant true
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST0]]{{.*}}outs([[INIT]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi1>) outs([[FILL]] : tensor<4xi1>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i1, %[[ARG2:[0-9a-zA-Z_]+]]: i1)
  // CHECK:   [[RES:%.+]] = arith.andi %[[ARG1]], %[[ARG2]] : i1
  // CHECK:   linalg.yield [[RES]] : i1
  // CHECK: tensor.expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xi1> into tensor<1x4xi1>
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  // CHECK: arith.constant false
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: or
  %1 = "tosa.reduce_any"(%arg0) {axis = 0 : i64} : (tensor<5x4xi1>) -> tensor<1x4xi1>

  return
}

// -----

// CHECK-LABEL: @concat
// CHECK-SAME: %[[ARG0:.+]]: tensor<5x1xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<6x1xf32>
func.func @concat(%arg0: tensor<5x1xf32>, %arg1: tensor<6x1xf32>) -> () {
  // CHECK: [[AXIS:%.+]] = arith.constant 0
  // CHECK: [[STRIDE:%.+]]   = arith.constant 1
  // CHECK: [[OFFSET:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX0:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX1:%.+]] = arith.constant 1 : index
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<11x1xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST]]{{.*}}outs([[INIT]]
  // CHECK: [[INSERT0:%.+]] = tensor.insert_slice %[[ARG0]] into [[FILL]][0, 0] [5, 1] [1, 1]
  // CHECK: [[INSERT1:%.+]] = tensor.insert_slice %[[ARG1]] into [[INSERT0]][5, 0] [6, 1] [1, 1]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<5x1xf32>, tensor<6x1xf32>)  -> (tensor<11x1xf32>)

  // CHECK: [[AXIS:%.+]] = arith.constant 1
  // CHECK: [[STRIDE:%.+]]   = arith.constant 1
  // CHECK: [[OFFSET:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX0:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX1:%.+]] = arith.constant 1 : index
  // CHECK: [[INIT:%.+]] = tensor.empty() : tensor<5x2xf32>
  // CHECK: [[CST:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill ins([[CST]]{{.*}}outs([[INIT]]
  // CHECK: [[INSERT0:%.+]] = tensor.insert_slice %[[ARG0]] into [[FILL]][0, 0] [5, 1] [1, 1]
  // CHECK: [[INSERT1:%.+]] = tensor.insert_slice %[[ARG0]] into [[INSERT0]][0, 1] [5, 1] [1, 1]
  %1 = "tosa.concat"(%arg0, %arg0) { axis = 1 : i64} : (tensor<5x1xf32>, tensor<5x1xf32>)  -> (tensor<5x2xf32>)
  return
}

// -----

// CHECK-LABEL: @concat_non_axis_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @concat_non_axis_dyn(%arg0: tensor<5x?xf32>, %arg1: tensor<6x?xf32>) -> () {
  // CHECK: %[[AXIS:.+]] = arith.constant 0
  // CHECK: %[[STRIDE:.+]]   = arith.constant 1
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[IDX0:.+]] = arith.constant 0 : index
  // CHECK: %[[IDX1:.+]] = arith.constant 1 : index
  // CHECK: %[[SIZE:.+]] = tensor.dim %[[ARG0]], %[[IDX1]]
  // CHECK: %[[IDX1_2:.+]] = arith.constant 1 : index
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[IDX1_2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<11x?xf32>
  // CHECK: %[[CST:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 0] [5, %[[SIZE]]] [1, 1]
  // CHECK: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][5, 0] [6, %[[SIZE]]] [1, 1]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<5x?xf32>, tensor<6x?xf32>)  -> (tensor<11x?xf32>)
  return
}

// -----

// CHECK-LABEL: @concat_axis_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]: 
func.func @concat_axis_dyn(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> () {
  // CHECK: %[[AXIS:.+]] = arith.constant 0
  // CHECK: %[[STRIDE:.+]]   = arith.constant 1
  // CHECK: %[[OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[IDX0:.+]] = arith.constant 0 : index
  // CHECK: %[[SIZE:.+]] = tensor.dim %[[ARG0]], %[[IDX0]]
  // CHECK: %[[IDX0_2:.+]] = arith.constant 0 : index
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[IDX0_2]]
  // CHECK: %[[IDX1:.+]] = arith.constant 1 : index
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]]) : tensor<?x3xf32>
  // CHECK: %[[CST:.+]] = arith.constant 0.0
  // CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}}outs(%[[INIT]]
  // CHECK: %[[DYN1:.+]] = tensor.dim %[[ARG0]], %[[AXIS]]
  // CHECK: %[[INSERT0:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 0] [%[[DYN1]], 3] [1, 1]
  // CHECK: %[[SUM:.+]]  = arith.addi %[[OFFSET]], %[[DYN1]]
  // CHECK: %[[DYN2:.+]] = tensor.dim %[[ARG1]], %[[AXIS]]
  // CHECK: %[[INSERT1:.+]] = tensor.insert_slice %[[ARG1]] into %[[INSERT0]][%[[SUM]], 0] [%[[DYN2]], 3] [1, 1]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<?x3xf32>, tensor<?x3xf32>)  -> (tensor<?x3xf32>)
  return
}

// -----
// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @rescale_i8(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C17:%.+]] = arith.constant 17
  // CHECK: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = "tosa.apply_scale"([[IN_ZEROED]], [[C0]], [[C1]]) {double_round = false}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[MINLT:%.+]] = arith.cmpi slt, [[SCALED_ZEROED]], [[CMIN]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[LOWER:%.+]] = arith.select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)

  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xi8>) outs([[INIT]] : tensor<2xui8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: ui8):
  // CHECK: [[C17:%.+]] = arith.constant 17
  // CHECK: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = "tosa.apply_scale"([[IN_ZEROED]], [[C0]], [[C1]]) {double_round = false}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant 0
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 255
  // CHECK-DAG: [[MINLT:%.+]] = arith.cmpi slt, [[SCALED_ZEROED]], [[CMIN]]
  // CHECK-DAG: [[LOWER:%.+]] = arith.select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: [[CAST:%.+]] = builtin.unrealized_conversion_cast [[TRUNC]] : i8 to ui8
  // CHECK: linalg.yield [[CAST]]
  %1 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xui8>)

  // CHECK: return
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @rescale_i8_dyn_batch
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @rescale_i8_dyn_batch(%arg0 : tensor<?x2xi8>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x2xi8>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x2xi8>) outs(%[[INIT]] : tensor<?x2xi8>)
  %0 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<?x2xi8>)  -> (tensor<?x2xi8>)

  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x2xui8>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x2xi8>) outs(%[[INIT]] : tensor<?x2xui8>)
  %1 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<?x2xi8>)  -> (tensor<?x2xui8>)

  return
}

// -----

// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @rescale_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @rescale_dyn(%arg0 : tensor<1x?x?x32xi32>) -> () {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DIM1]], %[[DIM2]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<1x?x?x32xi32>) outs(%[[INIT]] : tensor<1x?x?x32xi8>)
  %0 = "tosa.rescale"(%arg0) {double_round = true, input_zp = 0 : i32, multiplier = [1376784203 : i32], output_zp = 0 : i32, per_channel = false, scale32 = true, shift = [38 : i32]} : (tensor<1x?x?x32xi32>) -> tensor<1x?x?x32xi8>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_ui8
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @rescale_ui8(%arg0 : tensor<2xui8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<2xui8>) outs([[INIT]] : tensor<2xi8>)
  // CHECK: ^bb0([[IN:%.+]]: ui8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C17:%.+]] = arith.constant 17
  // CHECK: [[C22:%.+]] = arith.constant 22
  // CHECK-DAG: [[CAST:%.+]] = builtin.unrealized_conversion_cast [[IN]] : ui8 to i8
  // CHECK-DAG: [[IN32:%.+]] = arith.extui [[CAST]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C17]]
  // CHECK-DAG: [[SCALED:%.+]] = "tosa.apply_scale"([[IN_ZEROED]], [[C0]], [[C1]]) {double_round = false}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C22]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[MINLT:%.+]] = arith.cmpi slt, [[SCALED_ZEROED]], [[CMIN]]
  // CHECK-DAG: [[LOWER:%.+]] = arith.select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xui8>)  -> (tensor<2xi8>)

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_per_channel
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @rescale_per_channel(%arg0 : tensor<3xi8>) -> (tensor<3xi8>) {
  // CHECK: [[MULTIPLIERS:%.+]] = arith.constant dense<[42, 43, 0]>
  // CHECK: [[SHIFTS:%.+]] = arith.constant dense<[14, 15, 0]>
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%[[ARG0]], [[MULTIPLIERS]], [[SHIFTS]] : tensor<3xi8>, tensor<3xi32>, tensor<3xi8>) outs([[INIT]] : tensor<3xi8>)
  // CHECK: ^bb0([[IN:%.+]]: i8, [[MULTIPLIER:%.+]]: i32, [[SHIFT:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK: [[C243:%.+]] = arith.constant 243
  // CHECK: [[C252:%.+]] = arith.constant 252

  // CHECK-DAG: [[IN32:%.+]] = arith.extsi [[IN]]
  // CHECK-DAG: [[IN_ZEROED:%.+]] = arith.subi [[IN32]], [[C243]]
  // CHECK-DAG: [[SCALED:%.+]] = "tosa.apply_scale"([[IN_ZEROED]], [[MULTIPLIER]], [[SHIFT]]) {double_round = false}
  // CHECK-DAG: [[SCALED_ZEROED:%.+]] = arith.addi [[SCALED]], [[C252]]
  // CHECK-DAG: [[CMIN:%.+]] = arith.constant -128
  // CHECK-DAG: [[CMAX:%.+]] = arith.constant 127
  // CHECK-DAG: [[MINLT:%.+]] = arith.cmpi slt, [[SCALED_ZEROED]], [[CMIN]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[LOWER:%.+]] = arith.select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = arith.select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [42 : i32, 43 : i32, 44 : i32], shift = [14 : i32, 15 : i32, 64 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<3xi8>)  -> (tensor<3xi8>)

  // CHECK: return [[GENERIC]]
  return %0 : tensor<3xi8>
}

// -----

// CHECK-LABEL: @rescaleDoubleRound
func.func @rescaleDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = true}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [33 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: @rescaleUnnecessaryDoubleRound
func.func @rescaleUnnecessaryDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = false}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)
  return %0 : tensor<2xi8>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @reverse
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @reverse(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[RDIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel", "parallel"]} outs(%[[INIT]] : tensor<5x4xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[I1:.+]] = linalg.index 1
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[RDIM]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I0]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[READ_DIM]], %[[I1]]] : tensor<5x4xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>

  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[RDIM:.+]] = tensor.dim %[[ARG0]], %[[C1]]
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel", "parallel"]} outs(%[[INIT]] : tensor<5x4xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[I1:.+]] = linalg.index 1
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[RDIM]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I1]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[I0]], %[[READ_DIM]]] : tensor<5x4xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %1 = "tosa.reverse"(%arg0) {axis = 1 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @reverse_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @reverse_dyn(%arg0: tensor<?xi32>) -> () {
  // CHECK: %[[C0_1:.+]] = arith.constant 0
  // CHECK: %[[D0_1:.+]] = tensor.dim %[[ARG0]], %[[C0_1]]
  // CHECK: %[[C0_2:.+]] = arith.constant 0
  // CHECK: %[[D0_2:.+]] = tensor.dim %[[ARG0]], %[[C0_2]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[D0_1]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]]], iterator_types = ["parallel"]} outs(%[[INIT]] : tensor<?xi32>)
  // CHECK-DAG:   %[[I0:.+]] = linalg.index 0
  // CHECK-DAG:   %[[SUB1:.+]] = arith.constant 1
  // CHECK-DAG:   %[[RDIM_MINUS_C1:.+]] = arith.subi %[[D0_2]], %[[SUB1]]
  // CHECK-DAG:   %[[READ_DIM:.+]] = arith.subi %[[RDIM_MINUS_C1]], %[[I0]]
  // CHECK-DAG:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[READ_DIM]]] : tensor<?xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<?xi32>) -> tensor<?xi32>
  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x3xi8>
func.func @tile(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<2x2x1x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: tensor.collapse_shape [[GENERIC]] {{\[}}[0, 1, 2], [3]]
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1]} : (tensor<2x3xi8>)  -> (tensor<4x3xi8>)

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<1x2x2x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: tensor.collapse_shape [[GENERIC]] {{\[}}[0, 1], [2, 3]]
  %1 = "tosa.tile"(%arg0) {multiples = [1, 2]} : (tensor<2x3xi8>)  -> (tensor<2x6xi8>)

  // CHECK: [[INIT:%.+]] = tensor.empty()
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs([[INIT]] : tensor<5x2x7x3xi8>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i8
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: tensor.collapse_shape [[GENERIC]] {{\[}}[0, 1], [2, 3]]
  %2 = "tosa.tile"(%arg0) {multiples = [5, 7]} : (tensor<2x3xi8>)  -> (tensor<10x21xi8>)

  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile_dyn_input
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @tile_dyn_input(%arg0 : tensor<?x3xi8>) -> () {
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST0]] : tensor<?x3xi8>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<?x3xi8>) outs(%[[INIT]] : tensor<2x?x1x3xi8>)
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1, 2, 3]]
  // CHECK: tensor.expand_shape %[[COLLAPSED]] {{\[}}[0, 1]]
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1]} : (tensor<?x3xi8>)  -> (tensor<?x3xi8>)

  return
}

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @tile_dyn_multiples
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @tile_dyn_multiples(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: %[[CST1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST1]] : tensor<2x3xi8>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x3xi8>) outs(%[[INIT]] : tensor<2x2x?x3xi8>)
  // CHECK: ^bb0(%[[ARG1:.+]]: i8,
  // CHECK:   linalg.yield %[[ARG1]] : i8
  // CHECK: %[[COLLAPSED:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1, 2, 3]]
  // CHECK: tensor.expand_shape %[[COLLAPSED]] {{\[}}[0, 1]]
  %0 = "tosa.tile"(%arg0) {multiples = [2, -1]} : (tensor<2x3xi8>)  -> (tensor<2x?xi8>)

  return
}

// -----

// CHECK-LABEL: @pad_float
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]: 
func.func @pad_float(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<4x9xf32>)
  return %1 : tensor<4x9xf32>
}

func.func @pad_int(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 0 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

func.func @pad_quant(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 42 : i32
  // CHECK: tensor.pad
  // CHECK:   tensor.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0) {quantization_info = #tosa.pad_quant<input_zp = 42>} : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

// -----

func.func @pad_float_explicit(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 4.200000e+01 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = arith.constant dense<42.0> : tensor<f32>
  %2 = "tosa.pad"(%arg0, %0, %1)  : (tensor<1x2xf32>, tensor<2x2xi32>, tensor<f32>)  -> (tensor<4x9xf32>)
  return %2 : tensor<4x9xf32>
}

// -----

func.func @pad_dyn_input(%arg0 : tensor<?x2xf32>) -> (tensor<?x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<?x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<?x2xf32>, tensor<2x2xi32>)  -> (tensor<?x9xf32>)
  return %1 : tensor<?x9xf32>
}

func.func @pad_dyn_padding(%arg0 : tensor<1x2xf32>) -> (tensor<?x9xf32>) {
  %0 = arith.constant dense<[[-1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK-DAG: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK-DAG: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK-DAG: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: tensor.pad %[[ARG0]] low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK:   tensor.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<?x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<?x9xf32>)
  return %1 : tensor<?x9xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$MAP3:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP4:.*]] = affine_map<(d0) -> ()>

func.func @argmax(%arg0 : tensor<3x2xi32>, %arg1 : tensor<6xf32>) -> () {
  // CHECK: [[IDX_INIT:%.+]] = tensor.empty()
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill ins([[IDX_MIN]]{{.*}}outs([[IDX_INIT]]
  // CHECK: [[VAL_INIT:%.+]] = tensor.empty()
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill ins([[VAL_MIN]]{{.*}}outs([[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[ARG0]] : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<2xi32>, tensor<2xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   [[IDX:%.+]] = linalg.index 0
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_VAL:%.+]] = arith.select [[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_IDX:%.+]] = arith.select [[CMP]], [[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %0 = "tosa.argmax"(%arg0) { axis = 0 : i64} : (tensor<3x2xi32>)  -> (tensor<2xi32>)

  // CHECK: [[IDX_INIT:%.+]] = tensor.empty()
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill ins([[IDX_MIN]]{{.*}}outs([[IDX_INIT]]
  // CHECK: [[VAL_INIT:%.+]] = tensor.empty()
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill ins([[VAL_MIN]]{{.*}}outs([[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#map0, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%[[ARG0]] : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   [[IDX:%.+]] = linalg.index 1
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_VAL:%.+]] = arith.select [[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   [[SELECT_IDX:%.+]] = arith.select [[CMP]], [[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %1 = "tosa.argmax"(%arg0) { axis = 1 : i64} : (tensor<3x2xi32>)  -> (tensor<3xi32>)

  // CHECK: arith.constant -3.40282347E+38 : f32
  // CHECK: linalg.index
  // CHECK: arith.index_cast
  // CHECK: arith.cmpf ogt
  // CHECK: select
  // CHECK: select
  // CHECK: linalg.yield
  %2 = "tosa.argmax"(%arg1) { axis = 0 : i64} : (tensor<6xf32>)  -> (tensor<i32>)

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>

func.func @argmax_dyn_non_axis(%arg0 : tensor<3x?xi32>) -> () {
  // CHECK: %[[CST1:.+]] = arith.constant 1
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST1]]
  // CHECK: %[[IDX_INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[IDX_MIN:.+]] = arith.constant 0 : i32
  // CHECK: %[[IDX_FILL:.+]] = linalg.fill ins(%[[IDX_MIN]]{{.*}}outs(%[[IDX_INIT]]
  // CHECK: %[[VAL_INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[VAL_MIN:.+]] = arith.constant -2147483648
  // CHECK: %[[VAL_FILL:.+]] = linalg.fill ins(%[[VAL_MIN]]{{.*}}outs(%[[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[ARG0]] : tensor<3x?xi32>) outs(%[[IDX_FILL]], %[[VAL_FILL]] : tensor<?xi32>, tensor<?xi32>)
  // CHECK: ^bb0(%[[ARG1:[0-9a-zA-Z_]+]]: i32, %[[ARG2:[0-9a-zA-Z_]+]]: i32, %[[ARG3:[0-9a-zA-Z_]+]]: i32
  // CHECK:   %[[IDX:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK:   %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_VAL:.+]] = arith.select %[[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_IDX:.+]] = arith.select %[[CMP]], %[[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield %[[SELECT_IDX]], %[[SELECT_VAL]]
  %0 = "tosa.argmax"(%arg0) { axis = 0 : i64} : (tensor<3x?xi32>)  -> (tensor<?xi32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>

func.func @argmax_dyn_axis(%arg0 : tensor<3x?xi32>) -> () {
  // CHECK: %[[IDX_INIT:.+]] = tensor.empty()
  // CHECK: %[[IDX_MIN:.+]] = arith.constant 0 : i32
  // CHECK: %[[IDX_FILL:.+]] = linalg.fill ins(%[[IDX_MIN]]{{.*}}outs(%[[IDX_INIT]]
  // CHECK: %[[VAL_INIT:.+]] = tensor.empty()
  // CHECK: %[[VAL_MIN:.+]] = arith.constant -2147483648
  // CHECK: %[[VAL_FILL:.+]] = linalg.fill ins(%[[VAL_MIN]]{{.*}}outs(%[[VAL_INIT]]
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[ARG0]] : tensor<3x?xi32>) outs(%[[IDX_FILL]], %[[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK:   %[[IDX:.+]] = linalg.index 1
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK:   %[[CMP:.+]] = arith.cmpi sgt, %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_VAL:.+]] = arith.select %[[CMP]], %[[ARG1]], %[[ARG3]]
  // CHECK:   %[[SELECT_IDX:.+]] = arith.select %[[CMP]], %[[CAST]], %[[ARG2]]
  // CHECK:   linalg.yield %[[SELECT_IDX]], %[[SELECT_VAL]]
  %0 = "tosa.argmax"(%arg0) { axis = 1 : i64} : (tensor<3x?xi32>)  -> (tensor<3xi32>)
  return
}

// -----

// CHECK-LABEL: @gather_float
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xf32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<2x3x2xf32>, tensor<2x3xi32>)  -> (tensor<2x3x2xf32>)
  return
}

// -----

// CHECK-LABEL: @gather_float_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_float_dyn(%arg0: tensor<?x3x2xf32>, %arg1: tensor<?x3xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<?x3xi32>) outs(%[[INIT]] : tensor<?x3x2xf32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<?x3x2xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<?x3x2xf32>, tensor<?x3xi32>)  -> (tensor<?x3x2xf32>)
  return
}

// -----

// CHECK-LABEL: @gather_int
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]
func.func @gather_int(%arg0: tensor<2x3x2xi32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[ARG1]] : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xi32>)
  // CHECK: ^bb0(%[[BBARG0:.+]]: i32, %[[BBARG1:.+]]: i32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[BBARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG0]][%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<2x3x2xi32>, tensor<2x3xi32>)  -> (tensor<2x3x2xi32>)
  return
}

// -----

// CHECK-LABEL: @table8
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8(%arg0: tensor<6xi8>, %arg1: tensor<512xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi8>, tensor<512xi8>)  -> (tensor<6xi8>)
  return
}

// -----

// CHECK-LABEL: @table16
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table16(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi16>) outs(%[[INIT]] : tensor<6xi32>)
  // CHECK: ^bb0(%[[ARG2:.*]]: i16, %[[ARG3:.*]]: i32)
  // CHECK: %[[EXT_IN:.+]] = arith.extsi %[[ARG2]]
  // CHECK: %[[C32768:.+]] = arith.constant 32768
  // CHECK: %[[C7:.+]] = arith.constant 7
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[C127:.+]] = arith.constant 127
  // CHECK: %[[INADD:.+]] = arith.addi %[[EXT_IN]], %[[C32768]]
  // CHECK: %[[IDX:.+]] = arith.shrui %[[INADD]], %[[C7]]
  // CHECK: %[[FRACTION:.+]] = arith.andi %[[INADD]], %[[C127]]
  // CHECK: %[[IDXPLUS1:.+]] = arith.addi %[[IDX]], %[[C1]]
  // CHECK: %[[IDX_CAST:.+]] = arith.index_cast %[[IDX]]
  // CHECK: %[[IDXPLUS1_CAST:.+]] = arith.index_cast %[[IDXPLUS1]]
  // CHECK: %[[BASE:.+]] = tensor.extract %[[ARG1]][%[[IDX_CAST]]]
  // CHECK: %[[NEXT:.+]] = tensor.extract %[[ARG1]][%[[IDXPLUS1_CAST]]]
  // CHECK: %[[BASE_EXT:.+]] = arith.extsi %[[BASE]]
  // CHECK: %[[NEXT_EXT:.+]] = arith.extsi %[[NEXT]]
  // CHECK: %[[BASE_MUL:.+]] = arith.shli %[[BASE_EXT]], %[[C7]]
  // CHECK: %[[DIFF:.+]] = arith.subi %[[NEXT_EXT]], %[[BASE_EXT]]
  // CHECK: %[[DIFF_MUL:.+]] = arith.muli %[[DIFF]], %[[FRACTION]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[BASE_MUL]], %[[DIFF_MUL]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi16>, tensor<513xi16>)  -> (tensor<6xi32>)
  return
}

// -----

// CHECK-LABEL: @table8_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8_dyn(%arg0: tensor<?xi8>, %arg1: tensor<512xi8>) -> () {
  // CHECK: %[[CST0:.+]] = arith.constant 0
  // CHECK: %[[DYN:.+]] = tensor.dim %[[ARG0]], %[[CST0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[DYN]])
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<?xi8>) outs(%[[INIT]] : tensor<?xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<?xi8>, tensor<512xi8>)  -> (tensor<?xi8>)
  return
}

// -----

// CHECK-LABEL: @table8_dyn_table
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
// CHECK-SAME:  %[[ARG1:[0-9a-zA-Z_]*]]:
func.func @table8_dyn_table(%arg0: tensor<6xi8>, %arg1: tensor<?xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty()
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%[[ARG0]] : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %[[ARG1]][%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi8>, tensor<?xi8>)  -> (tensor<6xi8>)
  return
}

// -----

// CHECK-LABEL:  @resize_nearest_int
func.func @resize_nearest_int(%arg0: tensor<1x15x13x1xi8>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x23x179x1xi8>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK: %[[XY_MIN:.+]] = arith.constant 0
  // CHECK: %[[Y_MAX:.+]] = arith.constant 14
  // CHECK: %[[X_MAX:.+]] = arith.constant 12

  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.constant 11
  // CHECK: %[[SCALE_Y_D:.*]] = arith.constant 7
  // CHECK: %[[SCALE_X_N:.*]] = arith.constant 89
  // CHECK: %[[SCALE_X_D:.*]] = arith.constant 6
  // CHECK: %[[OFFSET_Y:.*]] = arith.constant 0
  // CHECK: %[[OFFSET_X:.*]] = arith.constant 0
  // CHECK: %[[BORDER_Y:.*]] = arith.constant 0
  // CHECK: %[[BORDER_X:.*]] = arith.constant 0

  // find the remainder and integer component of the target index.

  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[Y]], %[[SCALE_Y_D]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_Y:.*]] = arith.divui %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[I_X:.*]] = arith.divui %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Round to the nearest neighor.

  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[SCALE_Y_N_HALF:.*]] = arith.shrsi %[[SCALE_Y_N]], %[[ONE]]
  // CHECK: %[[SCALE_X_N_HALF:.*]] = arith.shrsi %[[SCALE_X_N]], %[[ONE]]
  // CHECK: %[[PRED_Y:.*]] = arith.cmpi sge, %[[D_Y]], %[[SCALE_Y_N_HALF]]
  // CHECK: %[[PRED_X:.*]] = arith.cmpi sge, %[[D_X]], %[[SCALE_X_N_HALF]]
  // CHECK: %[[VAL_37:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_38:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_39:.*]] = arith.addi %[[I_Y]], %[[VAL_37]]
  // CHECK: %[[VAL_40:.*]] = arith.addi %[[I_X]], %[[VAL_38]]

  // This section applies bound checking to be within the input image.

  // CHECK: %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_39]], %[[XY_MIN]]
  // CHECK: %[[VAL_42:.*]] = arith.select %[[VAL_41]], %[[XY_MIN]], %[[VAL_39]]
  // CHECK: %[[VAL_43:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[VAL_39]]
  // CHECK: %[[VAL_44:.*]] = arith.select %[[VAL_43]], %[[Y_MAX]], %[[VAL_42]]
  // CHECK: %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_40]], %[[XY_MIN]]
  // CHECK: %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[XY_MIN]], %[[VAL_40]]
  // CHECK: %[[VAL_47:.*]] = arith.cmpi slt, %[[X_MAX]], %[[VAL_40]]
  // CHECK: %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[X_MAX]], %[[VAL_46]]

  // Extract the nearest value using the computed indices.

  // CHECK: %[[IDY:.+]] = arith.index_cast %[[VAL_44]]
  // CHECK: %[[IDX:.+]] = arith.index_cast %[[VAL_48]]
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[IDY]], %[[IDX]], %[[IDX_3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "NEAREST_NEIGHBOR", scale = [11, 7, 89, 6], offset = [0, 0], border = [0, 0]} : (tensor<1x15x13x1xi8>) -> tensor<1x23x179x1xi8>
           return
}

// -----

// CHECK-LABEL:  @resize_bilinear_int
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @resize_bilinear_int(%arg0: tensor<1x19x19x1xi8>) {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x289x289x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK: %[[XY_MIN:.+]] = arith.constant 0
  // CHECK: %[[Y_MAX:.+]] = arith.constant 18
  // CHECK: %[[X_MAX:.+]] = arith.constant 18
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.constant 16
  // CHECK: %[[SCALE_Y_D:.*]] = arith.constant 1
  // CHECK: %[[SCALE_X_N:.*]] = arith.constant 16
  // CHECK: %[[SCALE_X_D:.*]] = arith.constant 1
  // CHECK: %[[OFFSET_Y:.*]] = arith.constant 0
  // CHECK: %[[OFFSET_X:.*]] = arith.constant 0
  // CHECK: %[[BORDER_Y:.*]] = arith.constant 0
  // CHECK: %[[BORDER_X:.*]] = arith.constant 0

  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[Y]], %[[SCALE_Y_D]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[X]], %[[SCALE_X_D]]
  // CHECK: %[[Y:.*]] = arith.addi %[[TEMP_Y]], %[[OFFSET_Y]]
  // CHECK: %[[X:.*]] = arith.addi %[[TEMP_X]], %[[OFFSET_X]]
  // CHECK: %[[I_Y:.*]] = arith.divui %[[Y]], %[[SCALE_Y_N]]
  // CHECK: %[[I_X:.*]] = arith.divui %[[X]], %[[SCALE_X_N]]
  // CHECK: %[[TEMP_Y:.*]] = arith.muli %[[I_Y]], %[[SCALE_Y_N]]
  // CHECK: %[[TEMP_X:.*]] = arith.muli %[[I_X]], %[[SCALE_X_N]]
  // CHECK: %[[D_Y:.*]] = arith.subi %[[Y]], %[[TEMP_Y]]
  // CHECK: %[[D_X:.*]] = arith.subi %[[X]], %[[TEMP_X]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]
  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // Extract each corner of the bilinear interpolation.

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YLOI]], %[[XHII]], %[[IDX_3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %[[ARG0]][%[[IDX_0]], %[[YHII]], %[[XHII]], %[[IDX_3]]]

  // CHECK: %[[XLOLO:.+]] = arith.extsi %[[LOLO]]
  // CHECK: %[[XLOHI:.+]] = arith.extsi %[[LOHI]]
  // CHECK: %[[XHILO:.+]] = arith.extsi %[[HILO]]
  // CHECK: %[[XHIHI:.+]] = arith.extsi %[[HIHI]]

  // Compute the bilinear interpolation.

  // CHECK: %[[NDX:.+]] = arith.subi %[[SCALE_X_N]], %[[D_X]]
  // CHECK: %[[WLOLO:.+]] = arith.muli %[[XLOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.muli %[[XLOHI]], %[[D_X]]
  // CHECK: %[[LO:.+]] = arith.addi %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.muli %[[XHILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.muli %[[XHIHI]], %[[D_X]]
  // CHECK: %[[HI:.+]] = arith.addi %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subi %[[SCALE_Y_N]], %[[D_Y]]
  // CHECK: %[[WLO:.+]] = arith.muli %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.muli %[[HI]], %[[D_Y]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round to the nearest index.
  %0 = "tosa.resize"(%arg0) {mode = "BILINEAR", scale = [16, 1, 16, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x19x19x1xi8>) -> tensor<1x289x289x1xi32>
           return
}

// -----

// CHECK-LABEL: @resize_nearest_fp
func.func @resize_nearest_fp(%input: tensor<1x50x48x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x1600x1536x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK: %[[XYMIN:.*]] = arith.constant 0
  // CHECK: %[[YMAX:.*]] = arith.constant 49
  // CHECK: %[[XMAX:.*]] = arith.constant 47
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX2]]
  // CHECK: %[[ISCALE_Y_N:.*]] = arith.constant 64
  // CHECK: %[[ISCALE_Y_D:.*]] = arith.constant 2
  // CHECK: %[[ISCALE_X_N:.*]] = arith.constant 64
  // CHECK: %[[ISCALE_X_D:.*]] = arith.constant 2
  // CHECK: %[[IOFFSET_Y:.*]] = arith.constant -31
  // CHECK: %[[IOFFSET_X:.*]] = arith.constant -31
  // CHECK: %[[IBORDER_Y:.*]] = arith.constant 31
  // CHECK: %[[IBORDER_X:.*]] = arith.constant 31

  // CHECK: %[[Y0:.+]] = arith.uitofp %[[Y]]
  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.uitofp %[[IOFFSET_Y]]
  // CHECK: %[[OFFSET_X:.*]] = arith.uitofp %[[IOFFSET_X]]

  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]

  // Find the remainder and integer component of the target index.

  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[VAL_39:.*]] = arith.fptosi %[[VAL_35]]
  // CHECK: %[[VAL_40:.*]] = arith.fptosi %[[VAL_36]]

  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01
  // CHECK: %[[PRED_Y:.*]] = arith.cmpf oge, %[[D_Y]], %[[HALF]]
  // CHECK: %[[PRED_X:.*]] = arith.cmpf oge, %[[D_X]], %[[HALF]]
  // CHECK: %[[ROUND_Y:.*]] = arith.select %[[PRED_Y]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[ROUND_X:.*]] = arith.select %[[PRED_X]], %[[ONE]], %[[ZERO]]
  // CHECK: %[[VAL_48:.*]] = arith.addi %[[VAL_39]], %[[ROUND_Y]]
  // CHECK: %[[VAL_49:.*]] = arith.addi %[[VAL_40]], %[[ROUND_X]]

  // CHECK: %[[VAL_50:.*]] = arith.cmpi slt, %[[VAL_48]], %[[XYMIN]]
  // CHECK: %[[VAL_51:.*]] = arith.select %[[VAL_50]], %[[XYMIN]], %[[VAL_48]]
  // CHECK: %[[VAL_52:.*]] = arith.cmpi slt, %[[YMAX]], %[[VAL_48]]
  // CHECK: %[[VAL_53:.*]] = arith.select %[[VAL_52]], %[[YMAX]], %[[VAL_51]]
  // CHECK: %[[VAL_54:.*]] = arith.cmpi slt, %[[VAL_49]], %[[XYMIN]]
  // CHECK: %[[VAL_55:.*]] = arith.select %[[VAL_54]], %[[XYMIN]], %[[VAL_49]]
  // CHECK: %[[VAL_56:.*]] = arith.cmpi slt, %[[XMAX]], %[[VAL_49]]
  // CHECK: %[[VAL_57:.*]] = arith.select %[[VAL_56]], %[[XMAX]], %[[VAL_55]]

  // CHECK: %[[IDY:.*]] = arith.index_cast %[[VAL_53]]
  // CHECK: %[[IDX:.*]] = arith.index_cast %[[VAL_57]]
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[IDY]], %[[IDX]], %[[IDX3]]]
  // CHECK: linalg.yield %[[EXTRACT]]

  %output = "tosa.resize"(%input) {mode = "NEAREST_NEIGHBOR", scale = [64, 2, 64, 2], offset = [-31, -31], border = [31, 31]} : (tensor<1x50x48x1xf32>) -> tensor<1x1600x1536x1xf32>

  return
}

// -----

// CHECK-LABEL: @resize_bilinear_fp
func.func @resize_bilinear_fp(%input: tensor<1x23x23x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = tensor.empty() : tensor<1x89x89x1xf32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX_0:.+]] = linalg.index 0
  // CHECK: %[[IDX_1:.+]] = linalg.index 1
  // CHECK: %[[IDX_2:.+]] = linalg.index 2
  // CHECK: %[[IDX_3:.+]] = linalg.index 3
  // CHECK: %[[XY_MIN:.*]] = arith.constant 0
  // CHECK: %[[Y_MAX:.*]] = arith.constant 22
  // CHECK: %[[X_MAX:.*]] = arith.constant 22
  // CHECK: %[[Y:.+]] = arith.index_cast %[[IDX_1]]
  // CHECK: %[[X:.+]] = arith.index_cast %[[IDX_2]]
  // CHECK: %[[ISCALE_Y_N:.*]] = arith.constant 4
  // CHECK: %[[ISCALE_Y_D:.*]] = arith.constant 1
  // CHECK: %[[ISCALE_X_N:.*]] = arith.constant 4
  // CHECK: %[[ISCALE_X_D:.*]] = arith.constant 1
  // CHECK: %[[IOFFSET_Y:.*]] = arith.constant 0
  // CHECK: %[[IOFFSET_X:.*]] = arith.constant 0
  // CHECK: %[[IBORDER_Y:.*]] = arith.constant 0
  // CHECK: %[[IBORDER_X:.*]] = arith.constant 0

  // CHECK: %[[Y0:.+]] = arith.uitofp %[[Y]]
  // CHECK: %[[X0:.+]] = arith.uitofp %[[X]]
  // CHECK: %[[SCALE_Y_N:.*]] = arith.uitofp %[[ISCALE_Y_N]]
  // CHECK: %[[SCALE_Y_D:.*]] = arith.uitofp %[[ISCALE_Y_D]]
  // CHECK: %[[SCALE_X_N:.*]] = arith.uitofp %[[ISCALE_X_N]]
  // CHECK: %[[SCALE_X_D:.*]] = arith.uitofp %[[ISCALE_X_D]]
  // CHECK: %[[OFFSET_Y:.*]] = arith.uitofp %[[IOFFSET_Y]]
  // CHECK: %[[OFFSET_X:.*]] = arith.uitofp %[[IOFFSET_X]]

  // CHECK: %[[VAL_29:.*]] = arith.mulf %[[Y0]], %[[SCALE_Y_D]]
  // CHECK: %[[VAL_30:.*]] = arith.mulf %[[X0]], %[[SCALE_X_D]]
  // CHECK: %[[VAL_31:.*]] = arith.addf %[[VAL_29]], %[[OFFSET_Y]]
  // CHECK: %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[OFFSET_X]]
  // CHECK: %[[VAL_33:.*]] = arith.divf %[[VAL_31]], %[[SCALE_Y_N]]
  // CHECK: %[[VAL_34:.*]] = arith.divf %[[VAL_32]], %[[SCALE_X_N]]

  // CHECK: %[[VAL_35:.*]] = math.floor %[[VAL_33]]
  // CHECK: %[[VAL_36:.*]] = math.floor %[[VAL_34]]
  // CHECK: %[[D_Y:.*]] = arith.subf %[[VAL_33]], %[[VAL_35]]
  // CHECK: %[[D_X:.*]] = arith.subf %[[VAL_34]], %[[VAL_36]]
  // CHECK: %[[I_Y:.*]] = arith.fptosi %[[VAL_35]]
  // CHECK: %[[I_X:.*]] = arith.fptosi %[[VAL_36]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[Y1:.*]] = arith.addi %[[I_Y]], %[[ONE]]
  // CHECK: %[[X1:.*]] = arith.addi %[[I_X]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_Y]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_Y]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[I_Y]]
  // CHECK: %[[YLO:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[Y1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[Y_MAX]], %[[Y1]]
  // CHECK: %[[YHI:.*]] = arith.select %[[PRED]], %[[Y_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[I_X]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[I_X]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[I_X]]
  // CHECK: %[[XLO:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X1]], %[[XY_MIN]]
  // CHECK: %[[BOUND:.*]] = arith.select %[[PRED]], %[[XY_MIN]], %[[X1]]
  // CHECK: %[[PRED:.*]] = arith.cmpi slt, %[[X_MAX]], %[[X1]]
  // CHECK: %[[XHI:.*]] = arith.select %[[PRED]], %[[X_MAX]], %[[BOUND]]

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YLOI]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YLOI]], %[[XHII]], %[[IDX_3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YHII]], %[[XLOI]], %[[IDX_3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %arg0[%[[IDX_0]], %[[YHII]], %[[XHII]], %[[IDX_3]]]

  // CHECK: %[[NDX:.+]] = arith.subf %[[SCALE_X_N]], %[[D_X]]
  // CHECK: %[[WLOLO:.+]] = arith.mulf %[[LOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.mulf %[[LOHI]], %[[D_X]]
  // CHECK: %[[LO:.+]] = arith.addf %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.mulf %[[HILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.mulf %[[HIHI]], %[[D_X]]
  // CHECK: %[[HI:.+]] = arith.addf %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subf %[[SCALE_Y_N]], %[[D_Y]]
  // CHECK: %[[WLO:.+]] = arith.mulf %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.mulf %[[HI]], %[[D_Y]]
  // CHECK: %[[RESULT:.+]] = arith.addf %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]

  // Round by bilinear interpolation
  %output = "tosa.resize"(%input) {mode = "BILINEAR", scale = [4, 1, 4, 1], offset = [0, 0], border = [0, 0]} : (tensor<1x23x23x1xf32>) -> tensor<1x89x89x1xf32>

  return
}

// -----

// CHECK-LABEL: @resize_dyn
// CHECK-SAME: (%[[ARG0:[0-9a-zA-Z_]*]]:
func.func @resize_dyn(%input: tensor<?x2x2x1xi8>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[BATCH:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[BATCH]]) : tensor<?x4x4x1xi32>
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  %output = "tosa.resize"(%input) { scale = [4, 2, 4, 2], offset = [-1, -1], border = [1, 1], mode = "BILINEAR" } : (tensor<?x2x2x1xi8>)  -> (tensor<?x4x4x1xi32>)
  return
}

// -----

// Regression test for using the wrong rank.

// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> ()>
// CHECK-LABEL: @select_fp32
func.func @select_fp32(%arg0: tensor<1x1x5x5xi1>, %arg1: tensor<1x12x5x5xf32>, %arg2: tensor<f32>) -> tensor<1x12x5x5xf32> {
  // CHECK: linalg.generic
  %0 = "tosa.select"(%arg0, %arg1, %arg2) : (tensor<1x1x5x5xi1>, tensor<1x12x5x5xf32>, tensor<f32>) -> tensor<1x12x5x5xf32>
  return %0 : tensor<1x12x5x5xf32>
}


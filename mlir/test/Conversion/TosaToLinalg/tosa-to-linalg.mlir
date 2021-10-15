// RUN: mlir-opt --split-input-file --tosa-to-linalg-on-tensors %s -verify-diagnostics -o -| FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>

// CHECK-LABEL: @test_abs
func @test_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [] : tensor<f32>
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = []} ins(%arg0 : tensor<f32>) outs([[INIT]] : tensor<f32>) {
  // CHECK: ^bb0(%arg1: f32, %arg2: f32):
  // CHECK:   [[ELEMENT:%.+]] = math.abs %arg1
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
  // CHECK:   [[ELEMENT:%.+]] = math.abs %arg1
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
  // CHECK:   [[ELEMENT:%.+]] = math.abs %arg1
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.abs"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_abs
func @test_abs(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[DIM]]]
  // CHECK: linalg.generic
  // CHECK: math.abs
  %0 = "tosa.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @test_abs_dyn
func @test_abs_dyn(%arg0: tensor<2x?xf32>) -> tensor<2x?xf32> {
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [2, %[[DIM]]]
  // CHECK: linalg.generic
  // CHECK: math.abs
  %0 = "tosa.abs"(%arg0) : (tensor<2x?xf32>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>
}
// -----


// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> ()>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @test_broadcast
func @test_broadcast(%arg0: tensor<1xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2] : tensor<2xf32>
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_collapse_shape %arg0
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel"]} ins([[RESHAPE]], %arg1 : tensor<f32>, tensor<2xf32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %arg2, %arg3 : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> ()>

// CHECK-LABEL: @test_broadcast_swapped_args
func @test_broadcast_swapped_args(%arg0: tensor<2xf32>, %arg1: tensor<1xf32>) -> tensor<2xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2] : tensor<2xf32>
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_collapse_shape %arg1
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0, [[RESHAPE]] : tensor<2xf32>, tensor<f32>) outs([[INIT]] : tensor<2xf32>) {
  // CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %arg2, %arg3 : f32
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
func @test_multibroadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<2x1xf32>) -> tensor<2x3xf32> {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 3] : tensor<2x3xf32>
  // CHECK: [[RESHAPE1:%.+]] = linalg.tensor_collapse_shape %arg0 {{\[}}[0, 1]]
  // CHECK: [[RESHAPE2:%.+]] = linalg.tensor_collapse_shape %arg1 {{\[}}[0, 1]]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP1]], #[[$MAP2]], #[[$MAP0]]], iterator_types = ["parallel", "parallel"]} ins([[RESHAPE1]], [[RESHAPE2]] : tensor<3xf32>, tensor<2xf32>) outs([[INIT]] : tensor<2x3xf32>) {
  // CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  // CHECK:   [[ELEMENT:%.+]] = arith.addf %arg2, %arg3 : f32
  // CHECK:   linalg.yield [[ELEMENT]] : f32
  // CHECK: } -> tensor<2x3xf32>
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_simple_f32
func @test_simple_f32(%arg0: tensor<1xf32>) -> () {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = "tosa.tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: math.abs
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
  // CHECK: arith.cmpf
  // CHECK: select
  %14 = "tosa.maximum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  // CHECK: select
  %15 = "tosa.minimum"(%0, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: ceil
  %16 = "tosa.ceil"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: floor
  %17 = "tosa.floor"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  // CHECK: select
  %18 = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 5 : i64, min_fp = 1.0 : f32, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.cmpf
  // CHECK: select
  %19 = "tosa.reluN"(%0) {max_int = 5 : i64, max_fp = 5.0 : f32} : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: exp
  // CHECK: arith.addf
  // CHECK: arith.divf
  %20 = "tosa.sigmoid"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0.000000e+00
  // CHECK: arith.constant 5.000000e-01
  // CHECK: arith.constant -2.14748365E+9
  // CHECK: arith.constant 2.14748365E+9
  // CHECK: arith.addf
  // CHECK: arith.subf
  // CHECK: arith.cmpf olt
  // CHECK: select
  // CHECK: arith.cmpf olt
  // CHECK: select
  // CHECK: arith.cmpf olt
  // CHECK: select
  // CHECK: arith.fptosi
  %21 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpf
  %22 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.truncf
  %23 = "tosa.cast"(%0) : (tensor<1xf32>) -> tensor<1xf16>

  // CHECK: linalg.generic
  // CHECK: arith.divf
  %24 = "tosa.reciprocal"(%0) : (tensor<1xf32>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_f16
func @test_simple_f16(%arg0: tensor<1xf16>) -> () {

  // CHECK: linalg.generic
  // CHECK: arith.extf
  %0 = "tosa.cast"(%arg0) : (tensor<1xf16>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_simple_i16
func @test_simple_i16(%arg0: tensor<1xi16>) -> () {
  // CHECK: linalg.generic
  // CHECK: arith.extsi
  // CHECK: arith.extsi
  // CHECK: arith.muli
  %0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<1xi16>, tensor<1xi16>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func @test_simple_ui8(%arg0: tensor<1xui8>) -> () {
  // CHECK: arith.uitofp
  %0 = "tosa.cast"(%arg0) : (tensor<1xui8>) -> tensor<1xf32>
  return
}

// -----

// CHECK-LABEL: @test_simple_i32
func @test_simple_i32(%arg0: tensor<1xi32>) -> () {
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
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: arith.subi [[ZERO]], %arg1
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

  // CHECK: scf.while
  // CHECK: arith.cmpi ne
  // CHECK: scf.condition
  // CHECK: arith.shrui
  // CHECK: arith.subi
  // CHECK: scf.yield
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
  // CHECK: arith.cmpi
  // CHECK: select
  %20 = "tosa.reluN"(%0) {max_int = 5 : i64, max_fp = 5.0 : f32} : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: linalg.generic
  // CHECK: arith.constant -32768
  // CHECK: arith.constant 32767
  // CHECK: arith.cmpi slt
  // CHECK: select
  // CHECK: arith.cmpi slt
  // CHECK: select
  // CHECK: arith.trunci
  %21 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi16>

  // CHECK: linalg.generic
  // CHECK: arith.extsi
  %22 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi64>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpi
  %23 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xi1>

  // CHECK: linalg.generic
  // CHECK: arith.sitofp
  %24 = "tosa.cast"(%0) : (tensor<1xi32>) -> tensor<1xf32>

  // CHECK: linalg.generic
  // CHECK: arith.constant 0
  // CHECK: arith.cmpi sgt
  // CHECK: arith.subi
  // CHECK: select
  %25 = "tosa.abs"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>

  return
}

// -----

// CHECK-LABEL: @test_simple_ui8
func @test_simple_ui8(%arg0: tensor<1xi8>) -> () {

  // CHECK: linalg.generic
  // CHECK: sitofp
  %0 = "tosa.cast"(%arg0) : (tensor<1xi8>) -> tensor<1xf32>

  return
}

// -----

// CHECK-LABEL: @test_i8
func @test_i8(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK-DAG: %[[C127:.+]] = arith.constant -127
  // CHECK-DAG: %[[C126:.+]] = arith.constant 126
  // CHECK-DAG: %[[CMP1:.+]] = arith.cmpi slt, %arg1, %[[C127]]
  // CHECK-DAG: %[[SEL1:.+]] = select %[[CMP1]], %[[C127]]
  // CHECK-DAG: %[[CMP2:.+]] = arith.cmpi slt, %[[C126]], %arg1
  // CHECK: %[[SEL2:.+]] = select %[[CMP2]], %[[C126]], %[[SEL1]]
  %0 = "tosa.clamp"(%arg0) {min_int = -127 : i64, max_int = 126 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK-DAG: %[[C128:.+]] = arith.constant -128
  // CHECK-DAG: %[[C127:.+]] = arith.constant 127
  // CHECK-DAG: %[[CMP1:.+]] = arith.cmpi slt, %arg1, %[[C128]]
  // CHECK-DAG: %[[SEL1:.+]] = select %[[CMP1]], %[[C128]]
  // CHECK-DAG: %[[CMP2:.+]] = arith.cmpi slt, %[[C127]], %arg1
  // CHECK: %[[SEL2:.+]] = select %[[CMP2]], %[[C127]], %[[SEL1]]
  %1 = "tosa.clamp"(%arg0) {min_int = -130 : i64, max_int = 130 : i64, min_fp = 0.0 : f32, max_fp = 0.0 : f32} : (tensor<1xi8>) -> tensor<1xi8>

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
func @test_negate_quantized(%arg0: tensor<1xi8>) -> () {
  // CHECK: linalg.generic
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[EXT:%.+]] = arith.extsi %arg1 : i8 to i16
  // CHECK: [[SUB:%.+]] = arith.subi [[ZERO]], [[EXT]]
  // CHECK: [[MIN:%.+]] = arith.constant -128
  // CHECK: [[MAX:%.+]] = arith.constant 127
  // CHECK: [[PRED1:%.+]] = arith.cmpi slt, [[SUB]], [[MIN]]
  // CHECK: [[LBOUND:%.+]] = select [[PRED1]], [[MIN]], [[SUB]]
  // CHECK: [[PRED2:%.+]] = arith.cmpi slt, [[MAX]], [[SUB]]
  // CHECK: [[UBOUND:%.+]] = select [[PRED2]], [[MAX]], [[LBOUND]]
  // CHECK: [[TRUNC:%.+]] = arith.trunci [[UBOUND]]
  // CHECK: linalg.yield [[TRUNC]]
  %0 = "tosa.negate"(%arg0) {quantization_info = { input_zp = 0 : i32, output_zp = 0 : i32}} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: [[EXT:%.+]] = arith.extsi %arg1 : i8 to i16
  %1 = "tosa.negate"(%arg0) {quantization_info = { input_zp = 32639 : i32, output_zp = 0 : i32}} : (tensor<1xi8>) -> tensor<1xi8>

  // CHECK: linalg.generic
  // CHECK: [[EXT:%.+]] = arith.extsi %arg1 : i8 to i32
  %2 = "tosa.negate"(%arg0) {quantization_info = { input_zp = 32640 : i32, output_zp = 0 : i32}} : (tensor<1xi8>) -> tensor<1xi8>

  return
}

// -----

// CHECK-LABEL: @test_reshape_downrank
func @test_reshape_downrank(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_collapse_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6]} : (tensor<2x3xf32>) -> tensor<6xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank
func @test_reshape_uprank(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE:%.+]] = linalg.tensor_expand_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank
func @test_reshape_samerank(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xf32>)
  // CHECK-NEXT: %[[RESHAPE1:.*]] = linalg.tensor_collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = linalg.tensor_expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D
func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  // CHECK: linalg.tensor_collapse_shape %arg0 {{\[}}[0, 1, 2], [3], [4, 5]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6, 5, 77]} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_identity
func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %0 = "tosa.identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tosa.identity"(%arg1) : (tensor<1xi32>) -> tensor<1xi32>

  // CHECK: return %arg0, %arg1
  return %0, %1 : tensor<1xf32>, tensor<1xi32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @test_transpose
// CHECK-SAME: ([[ARG0:%.+]]: tensor<1x2x3xi32>)
func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
  %0 = arith.constant dense<[1, 2, 0]> : tensor<3xi32>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 3, 1]
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
func @test_transpose_dyn(%arg0: tensor<1x?x3x4xi32>) -> () {
  %0 = arith.constant dense<[1, 3, 0, 2]> : tensor<4xi32>
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[DIM]], 4, 1, 3]
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
func @test_transpose_dyn_multiple(%arg0: tensor<?x?xf32>) -> () {
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM0:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C1:.+]] = arith.constant 1
  // CHECK: %[[DIM1:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[DIM1]], %[[DIM0]]]
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
func @reduce_float(%arg0: tensor<5x4xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [4]
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<4xf32>)
  // CHECK: ^bb0(%arg1: f32, %arg2: f32)
  // CHECK:   [[RES:%.+]] = arith.addf %arg1, %arg2 : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK: linalg.tensor_expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5]
  // CHECK: [[CST0:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xf32>) outs([[FILL]] : tensor<5xf32>)
  // CHECK: ^bb0(%arg1: f32, %arg2: f32)
  // CHECK:   [[RES:%.+]] = arith.addf %arg1, %arg2 : f32
  // CHECK:   linalg.yield [[RES]] : f32
  // CHECK: linalg.tensor_expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<5xf32> into tensor<5x1xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64} : (tensor<5x4xf32>) -> tensor<5x1xf32>

  // CHECK: arith.constant 1.0
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  %2 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant 3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.cmpf olt
  // CHECK: select
  %3 = "tosa.reduce_min"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>

  // CHECK: arith.constant -3.40282347E+38 : f32
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.cmpf ogt
  // CHECK: select
  %4 = "tosa.reduce_max"(%arg0) {axis = 0 : i64} : (tensor<5x4xf32>) -> tensor<1x4xf32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @reduce_int
// CHECK-SAME: [[ARG0:%.+]]: tensor<5x4xi32>
func @reduce_int(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [4]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<4xi32>)
  // CHECK: ^bb0(%arg1: i32, %arg2: i32)
  // CHECK:   [[RES:%.+]] = arith.addi %arg1, %arg2 : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK: linalg.tensor_expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<1x4xi32>

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP2]]], iterator_types = ["parallel", "reduction"]} ins([[ARG0]] : tensor<5x4xi32>) outs([[FILL]] : tensor<5xi32>)
  // CHECK: ^bb0(%arg1: i32, %arg2: i32)
  // CHECK:   [[RES:%.+]] = arith.addi %arg1, %arg2 : i32
  // CHECK:   linalg.yield [[RES]] : i32
  // CHECK: linalg.tensor_expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<5xi32> into tensor<5x1xi32>
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
func @reduce_bool(%arg0: tensor<5x4xi1>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [4]
  // CHECK: [[CST0:%.+]] = arith.constant true
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins([[ARG0]] : tensor<5x4xi1>) outs([[FILL]] : tensor<4xi1>)
  // CHECK: ^bb0(%arg1: i1, %arg2: i1)
  // CHECK:   [[RES:%.+]] = arith.andi %arg1, %arg2 : i1
  // CHECK:   linalg.yield [[RES]] : i1
  // CHECK: linalg.tensor_expand_shape [[GENERIC]] {{\[}}[0, 1]] : tensor<4xi1> into tensor<1x4xi1>
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
func @concat(%arg0: tensor<5x1xf32>, %arg1: tensor<6x1xf32>) -> () {
  // CHECK: [[AXIS:%.+]] = arith.constant 0
  // CHECK: [[STRIDE:%.+]]   = arith.constant 1
  // CHECK: [[OFFSET:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX0:%.+]] = arith.constant 0 : index
  // CHECK: [[ARG0_DIM0:%.+]] = tensor.dim %arg0, [[IDX0]]
  // CHECK: [[IDX1:%.+]] = arith.constant 1 : index
  // CHECK: [[ARG0_DIM1:%.+]] = tensor.dim %arg0, [[IDX1]]
  // CHECK: [[ARG1_AXIS:%.+]] = tensor.dim %arg1, [[AXIS]]
  // CHECK: [[RESULT_AXIS:%.+]] = arith.addi [[ARG0_DIM0]], [[ARG1_AXIS]]
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [11, 1]
  // CHECK: [[CST:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST]], [[INIT]])
  // CHECK: [[ARG0_DIM0:%.+]] = tensor.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT0:%.+]] = tensor.insert_slice %arg0 into [[FILL]]{{\[}}[[OFFSET]], [[OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  // CHECK: [[NEW_OFFSET:%.+]] = arith.addi [[OFFSET]], [[ARG0_DIM0]]
  // CHECK: [[ARG1_DIM0:%.+]] = tensor.dim %arg1, [[AXIS]]
  // CHECK: [[INSERT1:%.+]] = tensor.insert_slice %arg1 into [[INSERT0]]{{\[}}[[NEW_OFFSET]], [[OFFSET]]] {{\[}}[[ARG1_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  %0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<5x1xf32>, tensor<6x1xf32>)  -> (tensor<11x1xf32>)

  // CHECK: [[AXIS:%.+]] = arith.constant 1
  // CHECK: [[STRIDE:%.+]]   = arith.constant 1
  // CHECK: [[OFFSET:%.+]] = arith.constant 0 : index
  // CHECK: [[IDX0:%.+]] = arith.constant 0 : index
  // CHECK: [[ARG0_DIM0:%.+]] = tensor.dim %arg0, [[IDX0]]
  // CHECK: [[IDX1:%.+]] = arith.constant 1 : index
  // CHECK: [[ARG0_DIM1:%.+]] = tensor.dim %arg0, [[IDX1]]
  // CHECK: [[ARG1_AXIS:%.+]] = tensor.dim %arg0, [[AXIS]]
  // CHECK: [[RESULT_AXIS:%.+]] = arith.addi [[ARG0_DIM1]], [[ARG1_AXIS]]
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 2]
  // CHECK: [[CST:%.+]] = arith.constant 0.0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST]], [[INIT]])
  // CHECK: [[ARG0_DIM1:%.+]] = tensor.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT0:%.+]] = tensor.insert_slice %arg0 into [[FILL]]{{\[}}[[OFFSET]], [[OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG0_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  // CHECK: [[NEW_OFFSET:%.+]] = arith.addi [[OFFSET]], [[ARG0_DIM1]]
  // CHECK: [[ARG1_DIM1:%.+]] = tensor.dim %arg0, [[AXIS]]
  // CHECK: [[INSERT1:%.+]] = tensor.insert_slice %arg0 into [[INSERT0]]{{\[}}[[OFFSET]], [[NEW_OFFSET]]] {{\[}}[[ARG0_DIM0]], [[ARG1_DIM1]]] {{\[}}[[STRIDE]], [[STRIDE]]]
  %1 = "tosa.concat"(%arg0, %arg0) { axis = 1 : i64} : (tensor<5x1xf32>, tensor<5x1xf32>)  -> (tensor<5x2xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_i8
func @rescale_i8(%arg0 : tensor<2xi8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
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
  // CHECK-DAG: [[LOWER:%.+]] = select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)

  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xi8>) outs([[INIT]] : tensor<2xui8>)
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
  // CHECK-DAG: [[LOWER:%.+]] = select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: [[CAST:%.+]] = builtin.unrealized_conversion_cast [[TRUNC]] : i8 to ui8
  // CHECK: linalg.yield [[CAST]]
  %1 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xui8>)

  // CHECK: return
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_ui8
func @rescale_ui8(%arg0 : tensor<2xui8>) -> () {
  // CHECK: [[C0:%.+]] = arith.constant 19689
  // CHECK: [[C1:%.+]] = arith.constant 15
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0 : tensor<2xui8>) outs([[INIT]] : tensor<2xi8>)
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
  // CHECK-DAG: [[LOWER:%.+]] = select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[MAXLT:%.+]] = arith.cmpi slt, [[CMAX]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 17 : i32, output_zp = 22 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xui8>)  -> (tensor<2xi8>)

  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @rescale_per_channel
func @rescale_per_channel(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  // CHECK: [[MULTIPLIERS:%.+]] = arith.constant dense<[42, 43]>
  // CHECK: [[SHIFTS:%.+]] = arith.constant dense<[14, 15]>
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]], #[[$MAP0]]], iterator_types = ["parallel"]} ins(%arg0, [[MULTIPLIERS]], [[SHIFTS]] : tensor<2xi8>, tensor<2xi32>, tensor<2xi8>) outs([[INIT]] : tensor<2xi8>)
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
  // CHECK-DAG: [[LOWER:%.+]] = select [[MINLT]], [[CMIN]], [[SCALED_ZEROED]]
  // CHECK-DAG: [[BOUNDED:%.+]] = select [[MAXLT]], [[CMAX]], [[LOWER]]
  // CHECK-DAG: [[TRUNC:%.+]] = arith.trunci [[BOUNDED]]
  // CHECK-DAG: linalg.yield [[TRUNC]]
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [42 : i32, 43 : i32], shift = [14 : i32, 15 : i32], scale32 = false, double_round = false, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)

  // CHECK: return [[GENERIC]]
  return %0 : tensor<2xi8>
}

// -----

// CHECK-LABEL: @rescaleDoubleRound
func @rescaleDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = true}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [33 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: @rescaleUnnecessaryDoubleRound
func @rescaleUnnecessaryDoubleRound(%arg0 : tensor<2xi8>) -> (tensor<2xi8>) {
  // CHECK: linalg.generic
  // CHECK: "tosa.apply_scale"
  // CHECK-SAME:  {double_round = false}
  %0 = "tosa.rescale"(%arg0) {input_zp = 243 : i32, output_zp = 252 : i32, multiplier = [19689 : i32], shift = [15 : i32], scale32 = true, double_round = true, per_channel = false} : (tensor<2xi8>)  -> (tensor<2xi8>)
  return %0 : tensor<2xi8>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @reverse
func @reverse(%arg0: tensor<5x4xi32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[RDIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [5, 4]
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
  // CHECK: %[[RDIM:.+]] = tensor.dim %arg0, %[[C1]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [5, 4]
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
func @reverse_dyn(%arg0: tensor<?xi32>) -> () {
  // CHECK: %[[C0_1:.+]] = arith.constant 0
  // CHECK: %[[D0_1:.+]] = tensor.dim %arg0, %[[C0_1]]
  // CHECK: %[[C0_2:.+]] = arith.constant 0
  // CHECK: %[[D0_2:.+]] = tensor.dim %arg0, %[[C0_2]]
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[D0_1]]]
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
func @tile(%arg0 : tensor<2x3xi8>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [2, 2, 1, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3xi8>) outs([[INIT]] : tensor<2x2x1x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_collapse_shape [[GENERIC]] {{\[}}[0, 1, 2], [3]]
  %0 = "tosa.tile"(%arg0) {multiples = [2, 1]} : (tensor<2x3xi8>)  -> (tensor<4x3xi8>)

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 2, 2, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3xi8>) outs([[INIT]] : tensor<1x2x2x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_collapse_shape [[GENERIC]] {{\[}}[0, 1], [2, 3]]
  %1 = "tosa.tile"(%arg0) {multiples = [1, 2]} : (tensor<2x3xi8>)  -> (tensor<2x6xi8>)

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [5, 2, 7, 3]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3xi8>) outs([[INIT]] : tensor<5x2x7x3xi8>)
  // CHECK:   linalg.yield %arg1 : i8
  // CHECK: linalg.tensor_collapse_shape [[GENERIC]] {{\[}}[0, 1], [2, 3]]
  %2 = "tosa.tile"(%arg0) {multiples = [5, 7]} : (tensor<2x3xi8>)  -> (tensor<10x21xi8>)

  return
}

// -----


// CHECK-LABEL: @matmul
func @matmul(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: [[FILLED:%.+]] = linalg.fill([[C0]], [[INIT]]) : f32, tensor<1x5x6xf32> -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x6xf32>) outs([[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----


// CHECK-LABEL: @matmul_quantized
func @matmul_quantized(%arg0: tensor<1x5x3xi8>, %arg1: tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>) {
  // CHECK: [[C0:%.+]] = arith.constant 0
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: [[FILLED:%.+]] = linalg.fill([[C0]], [[INIT]]) : i32, tensor<1x5x6xi32> -> tensor<1x5x6xi32>
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: linalg.quantized_batch_matmul ins(%arg0, %arg1, [[ONE]], [[TWO]] : tensor<1x5x3xi8>, tensor<1x3x6xi8>, i32, i32) outs([[FILLED]] : tensor<1x5x6xi32>) -> tensor<1x5x6xi32>
  %0 = "tosa.matmul"(%arg0, %arg1) {quantization_info = {a_zp = 1 : i32, b_zp = 2 : i32}} : (tensor<1x5x3xi8>, tensor<1x3x6xi8>) -> (tensor<1x5x6xi32>)
  return %0 : tensor<1x5x6xi32>
}

// -----

// CHECK-LABEL: @matmul_dyn_batch
func @matmul_dyn_batch(%arg0: tensor<?x5x3xf32>, %arg1: tensor<?x3x6xf32>) -> (tensor<?x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C0_0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [%[[DIM]], 5, 6]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0_0]], %[[INIT]]) : f32, tensor<?x5x6xf32> -> tensor<?x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x5x3xf32>, tensor<?x3x6xf32>) outs(%[[FILLED]] : tensor<?x5x6xf32>) -> tensor<?x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<?x5x3xf32>, tensor<?x3x6xf32>)  -> (tensor<?x5x6xf32>)
  return %0 : tensor<?x5x6xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func @matmul_dyn_independent_dim(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x?xf32>) -> (tensor<1x5x?xf32>) {
  // CHECK: %[[C2:.+]] = arith.constant 2
  // CHECK: %[[DIM:.+]] = tensor.dim %arg1, %[[C2]]
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 5, %[[DIM]]]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0]], %[[INIT]]) : f32, tensor<1x5x?xf32> -> tensor<1x5x?xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x?xf32>) outs(%[[FILLED]] : tensor<1x5x?xf32>) -> tensor<1x5x?xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x?xf32>)  -> (tensor<1x5x?xf32>)
  return %0 : tensor<1x5x?xf32>
}

// -----

// CHECK-LABEL: @matmul_dyn_independent_dim
func @matmul_dyn_independent_dim(%arg0: tensor<1x5x?xf32>, %arg1: tensor<1x?x6xf32>) -> (tensor<1x5x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 5, 6]
  // CHECK: %[[FILLED:.+]] = linalg.fill(%[[C0]], %[[INIT]]) : f32, tensor<1x5x6xf32> -> tensor<1x5x6xf32>
  // CHECK: linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x?xf32>, tensor<1x?x6xf32>) outs(%[[FILLED]] : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x?xf32>, tensor<1x?x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: @fully_connected
func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[ZERO]], [[INITT]])
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [3, 6]
  // CHECK: [[TRANSPOSE:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<6x3xf32>) outs([[INITT]] : tensor<3x6xf32>) {
  // CHECK: ^bb0([[IN:%.+]]: f32, [[UNUSED:%.+]]: f32):
  // CHECK:   linalg.yield [[IN]] : f32
  // CHECK: [[INITB:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[MATMUL:%.+]] = linalg.matmul ins(%arg0, [[TRANSPOSE]] : tensor<5x3xf32>, tensor<3x6xf32>) outs([[FILL]] : tensor<5x6xf32>) -> tensor<5x6xf32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xf32>, tensor<5x6xf32>) outs([[INITB]] : tensor<5x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<5x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: @quantized_fully_connected
func @quantized_fully_connected(%arg0: tensor<5x3xi8>, %arg1: tensor<6x3xi8>, %arg2: tensor<6xi32>) -> (tensor<5x6xi32>) {
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ZERO:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[ZERO]], [[INITT]])
  // CHECK: [[PERM:%.+]] = arith.constant dense<[1, 0]>
  // CHECK: [[INITT:%.+]] = linalg.init_tensor [3, 6]
  // CHECK: [[TRANSPOSE:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<6x3xi8>) outs([[INITT]] : tensor<3x6xi8>) {
  // CHECK: ^bb0([[IN:%.+]]: i8, [[UNUSED:%.+]]: i8):
  // CHECK:   linalg.yield [[IN]] : i8
  // CHECK: [[INITB:%.+]] = linalg.init_tensor [5, 6]
  // CHECK: [[ONE:%.+]] = arith.constant 1
  // CHECK: [[TWO:%.+]] = arith.constant 2
  // CHECK: [[MATMUL:%.+]] = linalg.quantized_matmul ins(%arg0, [[TRANSPOSE]], [[ONE]], [[TWO]] : tensor<5x3xi8>, tensor<3x6xi8>, i32, i32) outs([[FILL]] : tensor<5x6xi32>) -> tensor<5x6xi32>
  // CHECK: [[ADDED:%.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, [[MATMUL]] : tensor<6xi32>, tensor<5x6xi32>) outs([[INITB]]
  // CHECK: ^bb0([[IN1:%.+]]: i32, [[IN2:%.+]]: i32, [[UNUSED:%.+]]: i32):
  // CHECK:   [[ADD:%.+]] = arith.addi
  // CHECK:   linalg.yield [[ADD]] : i32
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) {quantization_info = {input_zp = 1:i32, weight_zp = 2:i32}} : (tensor<5x3xi8>, tensor<6x3xi8>, tensor<6xi32>)  -> (tensor<5x6xi32>)
  return %0 : tensor<5x6xi32>
}

// -----

// CHECK-LABEL: @fully_connected_dyn
func @fully_connected_dyn(%arg0: tensor<?x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<?x6xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[INITT:.+]] = linalg.init_tensor [%[[DIM]], 6]
  // CHECK: %[[ZERO:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill(%[[ZERO]], %[[INITT]])
  // CHECK: %[[PERM:.+]] = arith.constant dense<[1, 0]>
  // CHECK: %[[INITT:.+]] = linalg.init_tensor [3, 6]
  // CHECK: %[[TRANSPOSE:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<6x3xf32>) outs(%[[INITT]] : tensor<3x6xf32>) {
  // CHECK: ^bb0(%[[IN:.+]]: f32, %[[UNUSED:.+]]: f32):
  // CHECK:   linalg.yield %[[IN]] : f32
  // CHECK: %[[INITB:.+]] = linalg.init_tensor [%[[DIM]], 6]
  // CHECK: %[[MATMUL:.+]] = linalg.matmul ins(%arg0, %[[TRANSPOSE]] : tensor<?x3xf32>, tensor<3x6xf32>) outs(%[[FILL]] : tensor<?x6xf32>) -> tensor<?x6xf32>
  // CHECK: %[[ADDED:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg2, %[[MATMUL]] : tensor<6xf32>, tensor<?x6xf32>) outs(%[[INITB]] : tensor<?x6xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
  // CHECK:   %[[ADD:.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield %[[ADD]] : f32

  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<?x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<?x6xf32>)
  return %0 : tensor<?x6xf32>
}

// -----

func @pad_float(%arg0 : tensor<1x2xf32>) -> (tensor<4x9xf32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // TODO: Output contains multiple "arith.constant 1 : index".
  // CHECK: [[INDEX1:%.+]] = arith.constant 1 : index
  // CHECK: [[INDEX2:%.+]] = arith.constant 2 : index
  // CHECK: [[INDEX3:%.+]] = arith.constant 3 : index
  // CHECK: [[INDEX4:%.+]] = arith.constant 4 : index
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: linalg.pad_tensor %arg0 low{{\[}}%{{.*}}, [[INDEX3]]] high{{\[}}[[INDEX2]], [[INDEX4]]]  {
  // CHECK: ^bb0(%arg1: index, %arg2: index):  // no predecessors
  // CHECK:   linalg.yield [[CST]]
  // CHECK: } : tensor<1x2xf32> to tensor<4x9xf32>
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xf32>, tensor<2x2xi32>)  -> (tensor<4x9xf32>)
  return %1 : tensor<4x9xf32>
}

func @pad_int(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 0 : i32
  // CHECK: linalg.pad_tensor
  // CHECK:   linalg.yield [[CST]]
  %1 = "tosa.pad"(%arg0, %0)  : (tensor<1x2xi32>, tensor<2x2xi32>)  -> (tensor<4x9xi32>)
  return %1 : tensor<4x9xi32>
}

func @pad_quant(%arg0 : tensor<1x2xi32>) -> (tensor<4x9xi32>) {
  %0 = arith.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: [[CST:%.+]] = arith.constant 42 : i32
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
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill([[IDX_MIN]], [[IDX_INIT]])
  // CHECK: [[VAL_INIT:%.+]] = linalg.init_tensor [2]
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill([[VAL_MIN]], [[VAL_INIT]])
  // CHECK: linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<2xi32>, tensor<2xi32>)
  // CHECK:   [[IDX:%.+]] = linalg.index 0
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %arg2, %arg4
  // CHECK:   [[SELECT_VAL:%.+]] = select [[CMP]], %arg2, %arg4
  // CHECK:   [[SELECT_IDX:%.+]] = select [[CMP]], [[CAST]], %arg3
  // CHECK:   linalg.yield [[SELECT_IDX]], [[SELECT_VAL]]
  %0 = "tosa.argmax"(%arg0) { axis = 0 : i64} : (tensor<3x2xi32>)  -> (tensor<2xi32>)

  // CHECK: [[IDX_INIT:%.+]] = linalg.init_tensor [3]
  // CHECK: [[IDX_MIN:%.+]] = arith.constant 0 : i32
  // CHECK: [[IDX_FILL:%.+]] = linalg.fill([[IDX_MIN]], [[IDX_INIT]])
  // CHECK: [[VAL_INIT:%.+]] = linalg.init_tensor [3]
  // CHECK: [[VAL_MIN:%.+]] = arith.constant -2147483648
  // CHECK: [[VAL_FILL:%.+]] = linalg.fill([[VAL_MIN]], [[VAL_INIT]])
  // CHECK: linalg.generic {indexing_maps = [#map0, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<3x2xi32>) outs([[IDX_FILL]], [[VAL_FILL]] : tensor<3xi32>, tensor<3xi32>)
  // CHECK:   [[IDX:%.+]] = linalg.index 1
  // CHECK:   [[CAST:%.+]] = arith.index_cast [[IDX]]
  // CHECK:   [[CMP:%.+]] = arith.cmpi sgt, %arg2, %arg4
  // CHECK:   [[SELECT_VAL:%.+]] = select [[CMP]], %arg2, %arg4
  // CHECK:   [[SELECT_IDX:%.+]] = select [[CMP]], [[CAST]], %arg3
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

// CHECK-LABEL: @gather_float
func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [2, 3, 2]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xf32>)
  // CHECK: ^bb0(%[[ARG0:.+]]: i32, %[[ARG1:.+]]: f32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xf32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<2x3x2xf32>, tensor<2x3xi32>)  -> (tensor<2x3x2xf32>)
  return
}

// CHECK-LABEL: @gather_int
func @gather_int(%arg0: tensor<2x3x2xi32>, %arg1: tensor<2x3xi32>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [2, 3, 2]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<2x3xi32>) outs(%[[INIT]] : tensor<2x3x2xi32>)
  // CHECK: ^bb0(%[[ARG0:.+]]: i32, %[[ARG1:.+]]: i32)
  // CHECK:   %[[IDX0:.+]] = linalg.index 0
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG0]]
  // CHECK:   %[[IDX2:.+]] = linalg.index 2
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[CAST]], %[[IDX2]]] : tensor<2x3x2xi32>
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.gather"(%arg0, %arg1)  : (tensor<2x3x2xi32>, tensor<2x3xi32>)  -> (tensor<2x3x2xi32>)
  return
}

// -----

// CHECK-LABEL: @table8
func @table8(%arg0: tensor<6xi8>, %arg1: tensor<512xi8>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<6xi8>) outs(%[[INIT]] : tensor<6xi8>)
  // CHECK: ^bb0(%[[ARG_IN:.+]]: i8, %[[ARG_INIT:.+]]: i8)
  // CHECK:   %[[CAST:.+]] = arith.index_cast %[[ARG_IN]]
  // CHECK:   %[[OFFSET:.+]] = arith.constant 128
  // CHECK:   %[[ADD:.+]] = arith.addi %[[CAST]], %[[OFFSET]]
  // CHECK:   %[[EXTRACT:.+]] = tensor.extract %arg1[%[[ADD]]]
  // CHECK:   linalg.yield %[[EXTRACT]]
  %0 = "tosa.table"(%arg0, %arg1)  : (tensor<6xi8>, tensor<512xi8>)  -> (tensor<6xi8>)
  return
}

// -----

// CHECK-LABEL: @table16
func @table16(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [6]
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<6xi16>) outs(%[[INIT]] : tensor<6xi32>)
  // CHECK: ^bb0(%arg2: i16, %arg3: i32)
  // CHECK: %[[EXT_IN:.+]] = arith.extsi %arg2
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
  // CHECK: %[[BASE:.+]] = tensor.extract %arg1[%[[IDX_CAST]]]
  // CHECK: %[[NEXT:.+]] = tensor.extract %arg1[%[[IDXPLUS1_CAST]]]
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

// CHECK-LABEL: @max_pool
func @max_pool(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38
  // CHECK-DAG: [[INIT:%.+]] = linalg.init_tensor [1, 4, 32, 62]
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill([[CONST]], [[INIT]])
  // CHECK-DAG: [[KERNEL:%.+]] = linalg.init_tensor [3, 3]
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, [[KERNEL]] : tensor<1x6x34x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x32x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x32x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_padded
func @max_pool_padded(%arg0: tensor<1x6x34x62xf32>) -> () {
  // CHECK-DAG: [[CONST:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[PAD:%.+]] = linalg.pad_tensor %arg0 low[0, 0, 0, 0] high[0, 0, 1, 0]
  // CHECK-DAG:   linalg.yield [[CONST]]
  // CHECK-DAG: [[INITVAL:%.+]] = arith.constant -3.40282347E+38 : f32
  // CHECK-DAG: [[INIT:%.+]] = linalg.init_tensor [1, 4, 33, 62]
  // CHECK-DAG: [[FILL:%.+]] = linalg.fill([[INITVAL]], [[INIT]])
  // CHECK-DAG: [[KERNEL:%.+]] = linalg.init_tensor [3, 3]
  // CHECK: linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x6x35x62xf32>, tensor<3x3xf32>) outs([[FILL]] : tensor<1x4x33x62xf32>)
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 1], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x4x33x62xf32>)
  return
}

// CHECK-LABEL: @max_pool_i8
func @max_pool_i8(%arg0: tensor<1x6x34x62xi8>) -> () {
  // CHECK: arith.constant -128
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi8>)  -> (tensor<1x4x32x62xi8>)
  return
}

// CHECK-LABEL: @max_pool_i16
func @max_pool_i16(%arg0: tensor<1x6x34x62xi16>) -> () {
  // CHECK: arith.constant -32768
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi16>)  -> (tensor<1x4x32x62xi16>)
  return
}

// CHECK-LABEL: @max_pool_i32
func @max_pool_i32(%arg0: tensor<1x6x34x62xi32>) -> () {
  // CHECK: arith.constant -2147483648
  // CHECK: linalg.pooling_nhwc_max
  %0 = "tosa.max_pool2d"(%arg0) {pad = [0, 0, 0, 0], kernel = [3, 3], stride = [1, 1]} : (tensor<1x6x34x62xi32>)  -> (tensor<1x4x32x62xi32>)
  return
}
// -----

// CHECK-LABEL: @avg_pool
func @avg_pool(%arg0: tensor<1x6x34x62xf32>) -> (tensor<1x5x33x62xf32>) {
  // Initial piece computes the sum of the pooling region, with appropriate padding.
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[PAD:%.+]] = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK: [[CONST:%.+]] = arith.constant 0
  // CHECK: [[POOLINIT:%.+]] = linalg.init_tensor [1, 5, 33, 62]
  // CHECK: [[FILL:%.+]] = linalg.fill([[CONST]], [[POOLINIT]])
  // CHECK: [[KERNEL:%.+]] = linalg.init_tensor [4, 4]
  // CHECK: [[POOL:%.+]] = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins([[PAD]], [[KERNEL]] : tensor<1x8x36x62xf32>, tensor<4x4xf32>) outs([[FILL]] : tensor<1x5x33x62xf32>)
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 33, 62]
  // CHECK: [[GENERIC:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins([[POOL]] : tensor<1x5x33x62xf32>) outs([[INIT]] : tensor<1x5x33x62xf32>)
  // CHECK:   [[ZERO:%.0]] = arith.constant 0
  // CHECK:   [[ONE:%.+]] = arith.constant 1
  // CHECK:   [[HEIGHT:%.+]] = arith.constant 4
  // CHECK:   [[WIDTH:%.+]] = arith.constant 32
  // CHECK:   [[IDX1:%.+]] = linalg.index 1
  // CHECK:   [[IDX2:%.+]] = linalg.index 2

  // The large block below computes what portion of the kernel is within non-padded input.
  // CHECK:   [[NY:%.+]] = arith.subi [[HEIGHT]], [[IDX1]]
  // CHECK:   [[NX:%.+]] = arith.subi [[WIDTH]], [[IDX2]]
  // CHECK:   [[KH:%.+]] = arith.constant 4
  // CHECK:   [[PAD0:%.+]] = arith.constant 1
  // CHECK:   [[SUBP0:%.+]] = arith.subi [[IDX1]], [[PAD0]]
  // CHECK:   [[P0CMP:%.+]] = arith.cmpi slt, [[SUBP0]], [[ZERO]]
  // CHECK:   [[SELP0:%.+]] = select [[P0CMP]], [[SUBP0]], [[ZERO]]
  // CHECK:   [[ADDP0:%.+]] = arith.addi [[KH]], [[SELP0]]
  // CHECK:   [[PAD1:%.+]] = arith.constant 1
  // CHECK:   [[SUBP1:%.+]] = arith.subi [[NY]], [[PAD1]]
  // CHECK:   [[P1CMP:%.+]] = arith.cmpi slt, [[SUBP1]], [[ZERO]]
  // CHECK:   [[SELP1:%.+]] = select [[P1CMP]], [[SUBP1]], [[ZERO]]
  // CHECK:   [[ADDP1:%.+]] = arith.addi [[ADDP0]], [[SELP1]]
  // CHECK:   [[YCMP:%.+]] = arith.cmpi slt, [[ADDP1]], [[ONE]]
  // CHECK:   [[YSEL:%.+]] = select [[YCMP]], [[ONE]], [[ADDP1]]
  // CHECK:   [[KW:%.+]] = arith.constant 4 : index
  // CHECK:   [[PAD2:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP2:%.+]] = arith.subi [[IDX2]], [[PAD2]]
  // CHECK:   [[P2CMP:%.+]] = arith.cmpi slt, [[SUBP2]], [[ZERO]]
  // CHECK:   [[SELP2:%.+]] = select [[P2CMP]], [[SUBP2]], [[ZERO]]
  // CHECK:   [[ADDP2:%.+]] = arith.addi [[KW]], [[SELP2]]
  // CHECK:   [[PAD3:%.+]] = arith.constant 1 : index
  // CHECK:   [[SUBP3:%.+]] = arith.subi [[NX]], [[PAD3]]
  // CHECK:   [[P3CMP:%.+]] = arith.cmpi slt, [[SUBP3]], [[ZERO]]
  // CHECK:   [[SELP3:%.+]] = select [[P3CMP]], [[SUBP3]], [[ZERO]]
  // CHECK:   [[ADDP3:%.+]] = arith.addi [[ADDP2]], [[SELP3]]
  // CHECK:   [[XCMP:%.+]] = arith.cmpi slt, [[ADDP3]], [[ONE]]
  // CHECK:   [[XSEL:%.+]] = select [[XCMP]], [[ONE]], [[ADDP3]]

  // Given the valid coverage of the pooling region, normalize the summation.
  // CHECK:   [[C:%.+]] = arith.muli [[YSEL]], [[XSEL]]
  // CHECK:   [[CI:%.+]] = arith.index_cast [[C]]
  // CHECK:   [[CF:%.+]] = arith.sitofp [[CI]]
  // CHECK:   [[RESULT:%.+]] = arith.divf %arg1, [[CF]]
  // CHECK:   linalg.yield [[RESULT]]
  %0 = "tosa.avg_pool2d"(%arg0) {pad = [1, 1, 1, 1], kernel = [4, 4], stride = [1, 1]} : (tensor<1x6x34x62xf32>)  -> (tensor<1x5x33x62xf32>)
  return %0 : tensor<1x5x33x62xf32>
}

// -----

// CHECK-LABEL: @avg_pool_i8
func @avg_pool_i8(%arg0 : tensor<1x128x128x2xi8>) -> () {

  // CHECK: linalg.pooling_nhwc_sum
  // CHECK: linalg.generic

  // CHECK: %[[INZP:.+]] = arith.constant -128
  // CHECK: %[[INZP_OFF:.+]] = arith.muli %{{.+}}, %[[INZP]]
  // CHECK: %[[OFFSETED:.+]] = arith.subi %arg1, %[[INZP_OFF]]
  // CHECK: %[[NUMERATOR:.+]] = arith.constant 1073741825
  // CHECK: %[[MULTIPLIER:.+]] = arith.divui %[[NUMERATOR]], %{{.+}}
  // CHECK: %[[SHIFT:.+]] = arith.constant 30
  // CHECK: %[[SCALE:.+]] = "tosa.apply_scale"(%{{.+}}, %[[MULTIPLIER]], %[[SHIFT]]) {double_round = false}
  // CHECK: %[[OUTZP:.+]] = arith.constant -128
  // CHECK: %[[OUT:.+]] = arith.addi %[[SCALE]], %[[OUTZP]]
  // CHECK: %[[MIN:.+]] = arith.constant -2147483648
  // CHECK: %[[MAX:.+]] = arith.constant 2147483647
  // CHECK: %[[CMP_MIN:.+]] = arith.cmpi slt, %[[OUT]], %[[MIN]]
  // CHECK: %[[CLMP_MIN:.+]] = select %[[CMP_MIN]], %[[MIN]], %[[OUT]]
  // CHECK: %[[CMP_MAX:.+]] = arith.cmpi slt, %[[MAX]], %[[OUT]]
  // CHECK: %[[CLMP_MAX:.+]] = select %[[CMP_MAX]], %[[MAX]], %[[CLMP_MIN]]
  // CHECK: linalg.yield %[[CLMP_MAX]]
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [4, 4], pad = [0, 0, 0, 0], quantization_info = {input_zp = -128 : i32, output_zp = -128 : i32}, stride = [4, 4]} : (tensor<1x128x128x2xi8>) -> tensor<1x32x32x2xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>

// CHECK-LABEL: @conv2d_f32
func @conv2d_f32(%input: tensor<1x49x42x27xf32>, %weights: tensor<28x3x3x27xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[W_IN:.+]] = linalg.init_tensor [3, 3, 27, 28]
  // CHECK: %[[W:.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<28x3x3x27xf32>) outs(%[[W_IN]] : tensor<3x3x27x28xf32>)
  // CHECK:   linalg.yield %arg3 : f32
  // CHECK: %[[M_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
  // CHECK: %[[CST:.+]] = arith.constant 0
  // CHECK: %[[FILL:.+]] = linalg.fill
  // CHECK: %[[B_IN:.+]] = linalg.init_tensor [1, 45, 40, 28]
  // CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %[[W]] : tensor<1x49x42x27xf32>, tensor<3x3x27x28xf32>) outs(%[[FILL]] : tensor<1x45x40x28xf32>)
  // CHECK: %[[B:.+]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %[[CONV]] : tensor<28xf32>, tensor<1x45x40x28xf32>) outs(%[[B_IN]] : tensor<1x45x40x28xf32>)
  // CHECK:   arith.addf
  // CHECK:   linalg.yield %7 : f32
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [2, 1]} : (tensor<1x49x42x27xf32>, tensor<28x3x3x27xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_padded_f32
func @conv2d_padded_f32(%input: tensor<1x47x40x28xf32>, %weights: tensor<28x3x3x28xf32>, %bias: tensor<28xf32>) -> () {
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   linalg.yield %[[C0]]
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [2, 1]} : (tensor<1x47x40x28xf32>, tensor<28x3x3x28xf32>, tensor<28xf32>)  -> (tensor<1x45x40x28xf32>)
  return
}

// -----

// CHECK-LABEL: @conv2d_quant
func @conv2d_quant(%arg0 : tensor<1x12x12x1xi8>, %arg1 : tensor<1024x3x3x1xi8>, %arg2 : tensor<1024xi32>) -> () {
  // CHECK:   %[[C22:.+]] = arith.constant -22
  // CHECK: linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   linalg.yield %[[C22]]
  // CHECK: linalg.conv_2d_nhwc_hwcf_q
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [1, 1, 1, 1], quantization_info = {input_zp = -22 : i32, weight_zp = 42 : i32}, stride = [1, 1]} : (tensor<1x12x12x1xi8>, tensor<1024x3x3x1xi8>, tensor<1024xi32>) -> tensor<1x12x12x1024xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv
func @depthwise_conv(%arg0 : tensor<1x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 5, 3, 11]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 5, 5, 33]
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv2D_nhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = linalg.tensor_collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<1x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_strides
func @depthwise_conv_strides(%arg0 : tensor<1x11x9x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 5, 5, 3, 11]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 5, 5, 33]
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv2D_nhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>) outs([[FILL]] : tensor<1x5x5x3x11xf32>)
  // CHECK: [[COLLAPSED:%.+]] = linalg.tensor_collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<33xf32>, tensor<1x5x5x33xf32>) outs([[OUT]] : tensor<1x5x5x33xf32>) {
  // CHECK: ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
  // CHECK:   [[ADD:%.+]] = arith.addf %arg3, %arg4 : f32
  // CHECK:   linalg.yield [[ADD]] : f32
  // CHECK: } -> tensor<1x5x5x33xf32>
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1] } : (tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant
func @depthwise_conv_quant(%arg0 : tensor<1x12x12x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[PADV:%.+]] = arith.constant -128
  // CHECK: [[PAD:%.+]] = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]
  // CHECK:   linalg.yield [[PADV]]

  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 12, 12, 4, 128]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 12, 12, 512]
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv2D_nhwc_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[PAD]], %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x12x12x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = linalg.tensor_collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x12x12x512xi32>) outs([[OUT]] : tensor<1x12x12x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x12x12x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], quantization_info = {input_zp = -128 : i32, weight_zp = 42 : i32}, stride = [1, 1], dilation = [1, 1] } : (tensor<1x12x12x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x12x12x512xi32>
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @depthwise_conv_quant_dilations
func @depthwise_conv_quant_dilations(%arg0 : tensor<1x14x14x4xi8>, %arg1 : tensor<3x3x4x128xi8>, %arg2 : tensor<512xi32>) -> () {
  // CHECK: [[INIT:%.+]] = linalg.init_tensor [1, 10, 10, 4, 128]
  // CHECK: [[CST0:%.+]] = arith.constant 0
  // CHECK: [[FILL:%.+]] = linalg.fill([[CST0]], [[INIT]])
  // CHECK: [[OUT:%.+]] = linalg.init_tensor [1, 10, 10, 512]
  // CHECK: [[C128:%.+]] = arith.constant -128
  // CHECK: [[C42:%.+]] = arith.constant 42
  // CHECK: [[DEPTH:%.+]] = linalg.depthwise_conv2D_nhwc_q {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1, [[C128]], [[C42]] : tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, i32, i32) outs([[FILL]] : tensor<1x10x10x4x128xi32>)
  // CHECK: [[COLLAPSED:%.+]] = linalg.tensor_collapse_shape [[DEPTH]] {{\[}}[0], [1], [2], [3, 4]]
  // CHECK: [[BIAS:%.+]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, [[COLLAPSED]] : tensor<512xi32>, tensor<1x10x10x512xi32>) outs([[OUT]] : tensor<1x10x10x512xi32>) {
  // CHECK: ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
  // CHECK:   [[ADD:%.+]] = arith.addi %arg3, %arg4 : i32
  // CHECK:   linalg.yield [[ADD]] : i32
  // CHECK: } -> tensor<1x10x10x512xi32>
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], quantization_info = {input_zp = -128 : i32, weight_zp = 42 : i32}, stride = [1, 1], dilation = [2, 2] } : (tensor<1x14x14x4xi8>, tensor<3x3x4x128xi8>, tensor<512xi32>)  -> tensor<1x10x10x512xi32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv
func @transpose_conv(%arg0 : tensor<1x12x12x2xf32>, %arg1 : tensor<4x3x3x2xf32>, %arg2 : tensor<4xf32>) -> () {
  // CHECK: linalg.pad_tensor %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0]
  // CHECK: linalg.conv_2d_nhwc_hwcf
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], out_shape = [1, 14, 14, 4], stride = [1, 1]} : (tensor<1x12x12x2xf32>, tensor<4x3x3x2xf32>, tensor<4xf32>) -> tensor<1x14x14x4xf32>
  return
}

// -----

// CHECK-LABEL: @transpose_conv_dilated
func @transpose_conv_dilated(%arg0 : tensor<1x12x12x2xf32>, %arg1 : tensor<4x3x3x2xf32>, %arg2 : tensor<4xf32>) -> () {
  // CHECK: [[PAD:%.+]] = linalg.pad_tensor %arg0 low[0, 4, 4, 0] high[0, 4, 4, 0]
  // CHECK: linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins([[PAD]], {{%.+}} : tensor<1x20x20x2xf32>, tensor<3x3x2x4xf32>)
  %0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [2, 2], out_pad = [0, 0], out_shape = [1, 16, 16, 4], stride = [1, 1]} : (tensor<1x12x12x2xf32>, tensor<4x3x3x2xf32>, tensor<4xf32>) -> tensor<1x16x16x4xf32>
  return
}


// -----

// CHECK-LABEL: @resize_nearest
func @resize_nearest(%input: tensor<1x2x2x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 4, 4, 1]
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XYMIN:.+]] = arith.constant 0
  // CHECK-DAG: %[[YMAX:.+]] = arith.constant 1
  // CHECK-DAG: %[[XMAX:.+]] = arith.constant 1
  // CHECK-DAG: %[[Y:.+]] = arith.index_cast %[[IDX1]]
  // CHECK-DAG: %[[X:.+]] = arith.index_cast %[[IDX2]]
  // CHECK-DAG: %[[STRIDEY:.+]] = arith.constant 5.000000e-01
  // CHECK-DAG: %[[STRIDEX:.+]] = arith.constant 5.000000e-01
  // CHECK-DAG: %[[OFFSETY:.+]] = arith.constant 1.000000e-01
  // CHECK-DAG: %[[OFFSETX:.+]] = arith.constant 2.000000e-01
  // CHECK-DAG: %[[VAL4:.+]] = arith.uitofp %[[Y]]
  // CHECK-DAG: %[[VAL5:.+]] = arith.uitofp %[[X]]
  // CHECK-DAG: %[[VAL6:.+]] = arith.mulf %[[VAL4]], %[[STRIDEY]]
  // CHECK-DAG: %[[VAL7:.+]] = arith.mulf %[[VAL5]], %[[STRIDEX]]
  // CHECK-DAG: %[[VAL8:.+]] = arith.addf %[[VAL6]], %[[OFFSETY]]
  // CHECK-DAG: %[[VAL9:.+]] = arith.addf %[[VAL7]], %[[OFFSETX]]

  // Find the remainder and integer component of the target index.

  // CHECK-DAG: %[[VAL10:.+]] = math.floor %[[VAL8]]
  // CHECK-DAG: %[[VAL11:.+]] = math.floor %[[VAL9]]
  // CHECK-DAG: %[[VAL12:.+]] = arith.subf %[[VAL8]], %[[VAL10]]
  // CHECK-DAG: %[[VAL13:.+]] = arith.subf %[[VAL9]], %[[VAL11]]
  // CHECK-DAG: %[[VAL14:.+]] = arith.fptosi %[[VAL10]]
  // CHECK-DAG: %[[VAL15:.+]] = arith.fptosi %[[VAL11]]

  // Round to the nearest index.

  // CHECK-DAG: %[[ROUND:.+]] = arith.constant 5.000000e-01
  // CHECK-DAG: %[[VAL16:.+]] = arith.cmpf oge, %[[VAL12]], %[[ROUND]]
  // CHECK-DAG: %[[VAL17:.+]] = arith.cmpf oge, %[[VAL13]], %[[ROUND]]
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.+]] = arith.constant 1
  // CHECK-DAG: %[[VAL18:.+]] = select %[[VAL16]], %[[ONE]], %[[ZERO]]
  // CHECK-DAG: %[[VAL19:.+]] = select %[[VAL17]], %[[ONE]], %[[ZERO]]
  // CHECK-DAG: %[[VAL20:.+]] = arith.addi %[[VAL14]], %[[VAL18]]
  // CHECK-DAG: %[[VAL21:.+]] = arith.addi %[[VAL15]], %[[VAL19]]

  // This section applies bound checking to be within the input image.

  // CHECK-DAG: %[[VAL22:.+]] = arith.cmpi slt, %[[VAL20]], %[[XYMIN]]
  // CHECK-DAG: %[[VAL23:.+]] = select %[[VAL22]], %[[XYMIN]], %[[VAL20]]
  // CHECK-DAG: %[[VAL24:.+]] = arith.cmpi slt, %[[YMAX]], %[[VAL20]]
  // CHECK-DAG: %[[VAL25:.+]] = select %[[VAL24]], %[[YMAX]], %[[VAL23]]
  // CHECK-DAG: %[[VAL26:.+]] = arith.cmpi slt, %[[VAL21]], %[[XYMIN]]
  // CHECK-DAG: %[[VAL27:.+]] = select %[[VAL26]], %[[XYMIN]], %[[VAL21]]
  // CHECK-DAG: %[[VAL28:.+]] = arith.cmpi slt, %[[XMAX]], %[[VAL21]]
  // CHECK-DAG: %[[VAL29:.+]] = select %[[VAL28]], %[[XMAX]], %[[VAL27]]

  // Extract the nearest value using the computed indices.

  // CHECK-DAG: %[[IDY:.+]] = arith.index_cast %[[VAL25]]
  // CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[VAL29]]
  // CHECK-DAG: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[IDY]], %[[IDX]], %[[IDX3]]]
  // CHECK: linalg.yield %[[EXTRACT]]
  %output = "tosa.resize"(%input) { output_size = [4, 4], stride = [0, 0], offset = [0, 0], stride_fp = [0.5 : f32, 0.5 : f32], offset_fp = [0.1 : f32, 0.2 : f32], shift = 0 : i32, mode = "NEAREST_NEIGHBOR" } : (tensor<1x2x2x1xf32>)  -> (tensor<1x4x4x1xf32>)

  return
}

// -----

// CHECK-LABEL: @resize_bilinear
func @resize_bilinear(%input: tensor<1x2x2x1xf32>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 4, 4, 1]
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK: %[[XYMIN:.+]] = arith.constant 0
  // CHECK: %[[YMAX:.+]] = arith.constant 1
  // CHECK: %[[XMAX:.+]] = arith.constant 1

  // CHECK: %[[VAL10:.+]] = math.floor %[[VAL8:.+]]
  // CHECK: %[[VAL11:.+]] = math.floor %[[VAL9:.+]]

  // CHECK: %[[DY:.+]] = arith.subf %[[VAL8:.+]], %[[VAL10]]
  // CHECK: %[[DX:.+]] = arith.subf %[[VAL9:.+]], %[[VAL11]]

  // CHECK: %[[Y0:.+]] = arith.fptosi %[[VAL10]]
  // CHECK: %[[X0:.+]] = arith.fptosi %[[VAL11]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK: %[[ONE:.+]] = arith.constant 1
  // CHECK: %[[Y1:.+]] = arith.addi %[[Y0]], %[[ONE]]
  // CHECK: %[[X1:.+]] = arith.addi %[[X0]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[Y0]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[Y0]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[YMAX]], %[[Y0]]
  // CHECK: %[[YLO:.+]] = select %[[PRED]], %[[YMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[Y1]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[Y1]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[YMAX]], %[[Y1]]
  // CHECK: %[[YHI:.+]] = select %[[PRED]], %[[YMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[X0]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[X0]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[XMAX]], %[[X0]]
  // CHECK: %[[XLO:.+]] = select %[[PRED]], %[[XMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[X1]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[X1]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[XMAX]], %[[X1]]
  // CHECK: %[[XHI:.+]] = select %[[PRED]], %[[XMAX]], %[[BOUND]]

  // Extract each corner of the bilinear interpolation.

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YLOI]], %[[XLOI]], %[[IDX3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YLOI]], %[[XHII]], %[[IDX3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YHII]], %[[XLOI]], %[[IDX3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YHII]], %[[XHII]], %[[IDX3]]]

  // Compute the bilinear interpolation.

  // CHECK: %[[ONE:.+]] = arith.constant 1.000000e+00
  // CHECK: %[[NDX:.+]] = arith.subf %[[ONE]], %[[DX]]
  // CHECK: %[[WLOLO:.+]] = arith.mulf %[[LOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.mulf %[[LOHI]], %[[DX]]
  // CHECK: %[[LO:.+]] = arith.addf %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.mulf %[[HILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.mulf %[[HIHI]], %[[DX]]
  // CHECK: %[[HI:.+]] = arith.addf %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subf %[[ONE]], %[[DY]]
  // CHECK: %[[WLO:.+]] = arith.mulf %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.mulf %[[HI]], %[[DY]]
  // CHECK: %[[RESULT:.+]] = arith.addf %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]
  %output = "tosa.resize"(%input) { output_size = [4, 4], stride = [0, 0], offset = [0, 0], stride_fp = [0.5 : f32, 0.5 : f32], offset_fp = [0.1 : f32, 0.2 : f32], shift = 0 : i32, mode = "BILINEAR" } : (tensor<1x2x2x1xf32>)  -> (tensor<1x4x4x1xf32>)
  return
}

// -----

// CHECK-LABEL: @resize_nearest_int
func @resize_nearest_int(%input: tensor<1x2x2x1xi32>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 4, 4, 1]
  // CHECK: %[[GENERIC:.+]] = linalg.generic
  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX1:.+]] = linalg.index 1
  // CHECK: %[[IDX2:.+]] = linalg.index 2
  // CHECK: %[[IDX3:.+]] = linalg.index 3
  // CHECK-DAG: %[[XYMIN:.+]] = arith.constant 0
  // CHECK-DAG: %[[YMAX:.+]] = arith.constant 1
  // CHECK-DAG: %[[XMAX:.+]] = arith.constant 1
  // CHECK-DAG: %[[Y:.+]] = arith.index_cast %[[IDX1]]
  // CHECK-DAG: %[[X:.+]] = arith.index_cast %[[IDX2]]
  // CHECK-DAG: %[[STRIDEY:.+]] = arith.constant 128
  // CHECK-DAG: %[[STRIDEX:.+]] = arith.constant 128
  // CHECK-DAG: %[[OFFSETY:.+]] = arith.constant 1
  // CHECK-DAG: %[[OFFSETX:.+]] = arith.constant 2
  // CHECK-DAG: %[[EIGHT:.+]] = arith.constant 8
  // CHECK-DAG: %[[VAL4:.+]] = arith.muli %[[Y]], %[[STRIDEY]]
  // CHECK-DAG: %[[VAL5:.+]] = arith.muli %[[X]], %[[STRIDEX]]
  // CHECK-DAG: %[[VAL6:.+]] = arith.addi %[[VAL4]], %[[OFFSETY]]
  // CHECK-DAG: %[[VAL7:.+]] = arith.addi %[[VAL5]], %[[OFFSETX]]

  // Find the remainder and integer component of the target index.


  // CHECK-DAG: %[[VAL8:.+]] = arith.shrsi %[[VAL6]], %[[EIGHT]]
  // CHECK-DAG: %[[VAL9:.+]] = arith.shrsi %[[VAL7]], %[[EIGHT]]
  // CHECK-DAG: %[[VAL10:.+]] = arith.shli %[[VAL8]], %[[EIGHT]]
  // CHECK-DAG: %[[VAL11:.+]] = arith.shli %[[VAL9]], %[[EIGHT]]
  // CHECK-DAG: %[[VAL12:.+]] = arith.subi %[[VAL6]], %[[VAL10]]
  // CHECK-DAG: %[[VAL13:.+]] = arith.subi %[[VAL7]], %[[VAL11]]

  // Round to the nearest index.

  // CHECK-DAG: %[[ROUND:.+]] = arith.constant 128
  // CHECK-DAG: %[[VAL16:.+]] = arith.cmpi sge, %[[VAL12]], %[[ROUND]]
  // CHECK-DAG: %[[VAL17:.+]] = arith.cmpi sge, %[[VAL13]], %[[ROUND]]
  // CHECK-DAG: %[[ZERO:.+]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.+]] = arith.constant 1
  // CHECK-DAG: %[[VAL18:.+]] = select %[[VAL16]], %[[ONE]], %[[ZERO]]
  // CHECK-DAG: %[[VAL19:.+]] = select %[[VAL17]], %[[ONE]], %[[ZERO]]
  // CHECK-DAG: %[[VAL20:.+]] = arith.addi %[[VAL8]], %[[VAL18]]
  // CHECK-DAG: %[[VAL21:.+]] = arith.addi %[[VAL9]], %[[VAL19]]

  // This section applies bound checking to be within the input image.

  // CHECK-DAG: %[[VAL22:.+]] = arith.cmpi slt, %[[VAL20]], %[[XYMIN]]
  // CHECK-DAG: %[[VAL23:.+]] = select %[[VAL22]], %[[XYMIN]], %[[VAL20]]
  // CHECK-DAG: %[[VAL24:.+]] = arith.cmpi slt, %[[YMAX]], %[[VAL20]]
  // CHECK-DAG: %[[VAL25:.+]] = select %[[VAL24]], %[[YMAX]], %[[VAL23]]
  // CHECK-DAG: %[[VAL26:.+]] = arith.cmpi slt, %[[VAL21]], %[[XYMIN]]
  // CHECK-DAG: %[[VAL27:.+]] = select %[[VAL26]], %[[XYMIN]], %[[VAL21]]
  // CHECK-DAG: %[[VAL28:.+]] = arith.cmpi slt, %[[XMAX]], %[[VAL21]]
  // CHECK-DAG: %[[VAL29:.+]] = select %[[VAL28]], %[[XMAX]], %[[VAL27]]

  // Extract the nearest value using the computed indices.

  // CHECK-DAG: %[[IDY:.+]] = arith.index_cast %[[VAL25]]
  // CHECK-DAG: %[[IDX:.+]] = arith.index_cast %[[VAL29]]
  // CHECK: %[[EXTRACT:.+]] = tensor.extract %arg0[%[[IDX0]], %[[IDY]], %[[IDX]], %[[IDX3]]]
  // CHECK: linalg.yield %[[EXTRACT]]
  %output = "tosa.resize"(%input) { output_size = [4, 4], stride = [128, 128], offset = [1, 2], stride_fp = [0. : f32, 0. : f32], offset_fp = [0. : f32, 0. : f32], shift = 8 : i32, mode = "NEAREST_NEIGHBOR" } : (tensor<1x2x2x1xi32>)  -> (tensor<1x4x4x1xi32>)
  return
}

// -----

// CHECK-LABEL: @resize_bilinear_int
func @resize_bilinear_int(%input: tensor<1x2x2x1xi8>) -> () {
  // CHECK: %[[INIT:.+]] = linalg.init_tensor [1, 4, 4, 1]
  // CHECK: %[[GENERIC:.+]] = linalg.generic

  // CHECK: %[[IDX0:.+]] = linalg.index 0
  // CHECK: %[[IDX3:.+]] = linalg.index 3

  // CHECK: %[[XYMIN:.+]] = arith.constant 0
  // CHECK: %[[YMAX:.+]] = arith.constant 1
  // CHECK: %[[XMAX:.+]] = arith.constant 1

  // CHECK: %[[Y0:.+]] = arith.shrsi
  // CHECK: %[[X0:.+]] = arith.shrsi
  // CHECK: %[[ROUNDY:.+]] = arith.shli %[[Y0]]
  // CHECK: %[[ROUNDX:.+]] = arith.shli %[[X0]]
  // CHECK: %[[DY:.+]] = arith.subi %10, %[[ROUNDY]]
  // CHECK: %[[DX:.+]] = arith.subi %11, %[[ROUNDX]]

  // Compute the left, right, and top indices for the bilinear interpolation.

  // CHECK: %[[ONE:.+]] = arith.constant 1
  // CHECK: %[[Y1:.+]] = arith.addi %[[Y0]], %[[ONE]]
  // CHECK: %[[X1:.+]] = arith.addi %[[X0]], %[[ONE]]

  // Bound check each dimension.

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[Y0]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[Y0]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[YMAX]], %[[Y0]]
  // CHECK: %[[YLO:.+]] = select %[[PRED]], %[[YMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[Y1]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[Y1]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[YMAX]], %[[Y1]]
  // CHECK: %[[YHI:.+]] = select %[[PRED]], %[[YMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[X0]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[X0]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[XMAX]], %[[X0]]
  // CHECK: %[[XLO:.+]] = select %[[PRED]], %[[XMAX]], %[[BOUND]]

  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[X1]], %[[XYMIN]]
  // CHECK: %[[BOUND:.+]] = select %[[PRED]], %[[XYMIN]], %[[X1]]
  // CHECK: %[[PRED:.+]] = arith.cmpi slt, %[[XMAX]], %[[X1]]
  // CHECK: %[[XHI:.+]] = select %[[PRED]], %[[XMAX]], %[[BOUND]]

  // Extract each corner of the bilinear interpolation.

  // CHECK: %[[YLOI:.+]] = arith.index_cast %[[YLO]]
  // CHECK: %[[YHII:.+]] = arith.index_cast %[[YHI]]
  // CHECK: %[[XLOI:.+]] = arith.index_cast %[[XLO]]
  // CHECK: %[[XHII:.+]] = arith.index_cast %[[XHI]]

  // CHECK: %[[LOLO:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YLOI]], %[[XLOI]], %[[IDX3]]]
  // CHECK: %[[LOHI:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YLOI]], %[[XHII]], %[[IDX3]]]
  // CHECK: %[[HILO:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YHII]], %[[XLOI]], %[[IDX3]]]
  // CHECK: %[[HIHI:.+]] = tensor.extract %arg0[%[[IDX0]], %[[YHII]], %[[XHII]], %[[IDX3]]]

  // CHECK: %[[XLOLO:.+]] = arith.extsi %[[LOLO]]
  // CHECK: %[[XLOHI:.+]] = arith.extsi %[[LOHI]]
  // CHECK: %[[XHILO:.+]] = arith.extsi %[[HILO]]
  // CHECK: %[[XHIHI:.+]] = arith.extsi %[[HIHI]]

  // Compute the bilinear interpolation.

  // CHECK: %[[SCALE:.+]] = arith.constant 256
  // CHECK: %[[NDX:.+]] = arith.subi %[[SCALE]], %[[DX]]
  // CHECK: %[[WLOLO:.+]] = arith.muli %[[XLOLO]], %[[NDX]]
  // CHECK: %[[WLOHI:.+]] = arith.muli %[[XLOHI]], %[[DX]]
  // CHECK: %[[LO:.+]] = arith.addi %[[WLOLO]], %[[WLOHI]]
  // CHECK: %[[WHILO:.+]] = arith.muli %[[XHILO]], %[[NDX]]
  // CHECK: %[[WHIHI:.+]] = arith.muli %[[XHIHI]], %[[DX]]
  // CHECK: %[[HI:.+]] = arith.addi %[[WHILO]], %[[WHIHI]]
  // CHECK: %[[NDY:.+]] = arith.subi %[[SCALE]], %[[DY]]
  // CHECK: %[[WLO:.+]] = arith.muli %[[LO]], %[[NDY]]
  // CHECK: %[[WHI:.+]] = arith.muli %[[HI]], %[[DY]]
  // CHECK: %[[RESULT:.+]] = arith.addi %[[WLO]], %[[WHI]]
  // CHECK: linalg.yield %[[RESULT]]
  %output = "tosa.resize"(%input) { output_size = [4, 4], stride = [128, 128], offset = [1, 2], stride_fp = [0. : f32, 0. : f32], offset_fp = [0. : f32, 0. : f32], shift = 8 : i32, mode = "BILINEAR" } : (tensor<1x2x2x1xi8>)  -> (tensor<1x4x4x1xi32>)
  return
}

// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.rewrite_as_constant
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @tensor_generate_constant(
//       CHECK:   %[[cst:.*]] = arith.constant dense<5.000000e+00> : tensor<2x3x5xf32>
//       CHECK:   return %[[cst]]
func.func @tensor_generate_constant() -> tensor<2x3x5xf32> {
  %cst = arith.constant 5.0 : f32
  %0 = tensor.generate {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
    tensor.yield %cst : f32
  } : tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

//         CHECK-LABEL: func @pad_of_ints(
//               CHECK: %[[cst:.*]] = arith.constant dense<[
// CHECK-SAME{LITERAL}:     [0, 0, 0, 0],
// CHECK-SAME{LITERAL}:     [0, 6, 7, 0],
// CHECK-SAME{LITERAL}:     [0, 8, 9, 0],
// CHECK-SAME{LITERAL}:     [0, 0, 0, 0]
// CHECK-SAME{LITERAL}:     ]> : tensor<4x4xi32>
//               CHECK: %[[cast:.*]] = tensor.cast %[[cst]] : tensor<4x4xi32> to tensor<?x?xi32>
//               CHECK: return %[[cast]]
func.func @pad_of_ints() -> tensor<?x?xi32> {
  %init = arith.constant dense<[[6, 7], [8, 9]]> : tensor<2x2xi32>
  %pad_value = arith.constant 0 : i32

  %c1 = arith.constant 1 : index

  %0 = tensor.pad %init low[%c1, %c1] high[%c1, %c1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : i32
  } : tensor<2x2xi32> to tensor<?x?xi32>

  return %0 : tensor<?x?xi32>
}

//         CHECK-LABEL: func @pad_of_floats(
//               CHECK: %[[cst:.*]] = arith.constant dense<[
// CHECK-SAME{LITERAL}:     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
// CHECK-SAME{LITERAL}:     [0.000000e+00, 6.000000e+00, 7.000000e+00, 0.000000e+00],
// CHECK-SAME{LITERAL}:     [0.000000e+00, 8.000000e+00, 9.000000e+00, 0.000000e+00],
// CHECK-SAME{LITERAL}:     [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]
// CHECK-SAME{LITERAL}:     ]> : tensor<4x4xf32>
//               CHECK: return %[[cst]]

func.func @pad_of_floats() -> tensor<4x4xf32> {
  %init = arith.constant dense<[[6.0, 7.0], [8.0, 9.0]]> : tensor<2x2xf32>
  %pad_value = arith.constant 0.0 : f32

  %0 = tensor.pad %init low[1, 1] high[1, 1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : f32
  } : tensor<2x2xf32> to tensor<4x4xf32>

  return %0 : tensor<4x4xf32>
}

//         CHECK-LABEL: func @pad_of_ints_no_low_dims(
//               CHECK: %[[cst:.*]] = arith.constant dense<[
// CHECK-SAME{LITERAL}:     [6, 7, 0],
// CHECK-SAME{LITERAL}:     [8, 9, 0],
// CHECK-SAME{LITERAL}:     [0, 0, 0]
// CHECK-SAME{LITERAL}:     ]> : tensor<3x3xi32>
//               CHECK: return %[[cst]]
func.func @pad_of_ints_no_low_dims() -> tensor<3x3xi32> {
  %init = arith.constant dense<[[6, 7], [8, 9]]> : tensor<2x2xi32>
  %pad_value = arith.constant 0 : i32

  %0 = tensor.pad %init low[0, 0] high[1, 1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : i32
  } : tensor<2x2xi32> to tensor<3x3xi32>

  return %0 : tensor<3x3xi32>
}

//         CHECK-LABEL: func @pad_of_ints_no_high_dims(
//               CHECK: %[[cst:.*]] = arith.constant dense<[
// CHECK-SAME{LITERAL}:     [0, 0, 0],
// CHECK-SAME{LITERAL}:     [0, 6, 7],
// CHECK-SAME{LITERAL}:     [0, 8, 9]
// CHECK-SAME{LITERAL}:     ]> : tensor<3x3xi32>
//               CHECK: return %[[cst]]
func.func @pad_of_ints_no_high_dims() -> tensor<3x3xi32> {
  %init = arith.constant dense<[[6, 7], [8, 9]]> : tensor<2x2xi32>
  %pad_value = arith.constant 0 : i32

  %0 = tensor.pad %init low[1, 1] high[0, 0] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : i32
  } : tensor<2x2xi32> to tensor<3x3xi32>

  return %0 : tensor<3x3xi32>
}

//         CHECK-LABEL: func @pad_multi_use_do_not_fold(
//               CHECK: %[[pad:.+]] = tensor.pad
//               CHECK: return %[[pad]]
func.func @pad_multi_use_do_not_fold() -> (tensor<?x?xi32>, tensor<2x2xi32>) {
  %init = arith.constant dense<[[6, 7], [8, 9]]> : tensor<2x2xi32>
  %pad_value = arith.constant 0 : i32

  %c1 = arith.constant 1 : index

  %0 = tensor.pad %init low[%c1, %c1] high[%c1, %c1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : i32
  } : tensor<2x2xi32> to tensor<?x?xi32>

  return %0, %init : tensor<?x?xi32>, tensor<2x2xi32>
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.tensor.rewrite_as_constant aggressive
    } : !transform.op<"func.func">
    transform.yield
  }
}

//         CHECK-LABEL: func @pad_aggressive_fold(
//               CHECK: %[[init:.*]] = arith.constant dense<7> : tensor<2x2xi32>
//               CHECK: %[[cst:.*]] = arith.constant dense<[
// CHECK-SAME{LITERAL}:     [0, 0, 0, 0],
// CHECK-SAME{LITERAL}:     [0, 7, 7, 0],
// CHECK-SAME{LITERAL}:     [0, 7, 7, 0],
// CHECK-SAME{LITERAL}:     [0, 0, 0, 0]
// CHECK-SAME{LITERAL}:     ]> : tensor<4x4xi32>
//               CHECK: %[[cast:.*]] = tensor.cast %[[cst]] : tensor<4x4xi32> to tensor<?x?xi32>
//               CHECK: return %[[cast]]
func.func @pad_aggressive_fold() -> (tensor<?x?xi32>, tensor<2x2xi32>) {
  %init = arith.constant dense<7> : tensor<2x2xi32>
  %pad_value = arith.constant 0 : i32

  %c1 = arith.constant 1 : index

  %0 = tensor.pad %init low[%c1, %c1] high[%c1, %c1] {
    ^bb0(%arg1: index, %arg2: index):
      tensor.yield %pad_value : i32
  } : tensor<2x2xi32> to tensor<?x?xi32>

  return %0, %init : tensor<?x?xi32>, tensor<2x2xi32>
}

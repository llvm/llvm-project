// RUN: mlir-opt -split-input-file -transform-interpreter %s | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%func_op: !transform.op<"func.func"> {transform.readonly}) {
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

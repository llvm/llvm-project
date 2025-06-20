// RUN: mlir-opt %s -transform-interpreter -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @create_dim(
//  CHECK-NEXT:   tensor.dim
func.func @create_dim() -> tensor<8x16xf32> {
  %s = arith.constant 1.0 : f32
  %t = tensor.splat %s : tensor<8x16xf32>
  return %t: tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %splat = transform.structured.match ops{["tensor.splat"]} in %module : (!transform.any_op) -> !transform.any_op
    %t = transform.get_operand %splat[0] : (!transform.any_op) -> !transform.any_value
    %_ = transform.tensor.get_dim %t[0] : (!transform.any_value) -> !transform.any_value
    transform.yield
  }
}

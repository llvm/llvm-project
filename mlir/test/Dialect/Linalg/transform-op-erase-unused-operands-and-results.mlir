// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
func.func @remove_deadargs_generic_basic(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 7.0 : f32
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = tensor.empty(%0) : tensor<?xf32>
  %2 = tensor.empty(%0) : tensor<?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types=["parallel"]} ins(%arg0, %1 : tensor<?xf32>, tensor<?xf32>) outs (%2:tensor<?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
    %4 = arith.addf  %arg1, %cst : f32
    linalg.yield %4 : f32
  } -> tensor<?xf32>
  return %3 : tensor<?xf32>
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.linalg.erase_unused_operands_and_results
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @remove_deadargs_generic_basic
//  CHECK-SAME: (%[[ARG0:.*]]: tensor<?xf32>) -> tensor<?xf32> {
//       CHECK: %[[GENERIC_OP:.*]] = linalg.generic
//  CHECK-SAME: ins(%[[ARG0]] : tensor<?xf32>)
//  CHECK-SAME: outs({{.*}} : tensor<?xf32>) {

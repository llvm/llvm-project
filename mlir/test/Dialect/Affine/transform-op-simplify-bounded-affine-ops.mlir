// RUN: mlir-opt  %s -allow-unregistered-dialect \
// RUN:     --transform-interpreter -verify-diagnostics \
// RUN:     --split-input-file | FileCheck %s

//     CHECK: func @simplify_min_max()
// CHECK-DAG:   %[[c50:.*]] = arith.constant 50 : index
// CHECK-DAG:   %[[c100:.*]] = arith.constant 100 : index
//     CHECK:   return %[[c50]], %[[c100]]
func.func @simplify_min_max() -> (index, index) {
  %0 = "test.some_op"() : () -> (index)
  %1 = affine.min affine_map<()[s0] -> (50, 100 - s0)>()[%0]
  %2 = affine.max affine_map<()[s0] -> (100, 80 + s0)>()[%0]
  return %1, %2 : index, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.some_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.simplify_bounded_affine_ops %0 with [%1 : !transform.any_op] within [0] and [20] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: func @simplify_min_sequence()
// CHECK:   %[[c1:.*]] = arith.constant 1 : index
// CHECK:   return %[[c1]]
func.func @simplify_min_sequence() -> index {
  %1 = "test.workgroup_id"() : () -> (index)
  %2 = affine.min affine_map<()[s0] -> (s0 * -32 + 1023, 32)>()[%1]
  %3 = "test.thread_id"() : () -> (index)
  %4 = affine.min affine_map<()[s0, s1] -> (s0 - s1 * (s0 ceildiv 32), s0 ceildiv 32)>()[%2, %3]
  return %4 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.min"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["test.workgroup_id"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.thread_id"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.simplify_bounded_affine_ops %0 with [%1, %2 : !transform.any_op, !transform.any_op] within [0, 0] and [31, 31] : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.min"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{incorrect number of lower bounds, expected 0 but found 1}}
    transform.affine.simplify_bounded_affine_ops %0 with [] within [0] and [] : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["affine.min"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error@+1 {{incorrect number of upper bounds, expected 0 but found 1}}
    transform.affine.simplify_bounded_affine_ops %0 with [] within [] and [5] : !transform.any_op
    transform.yield
  }
}

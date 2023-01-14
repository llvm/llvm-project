// RUN: mlir-opt  %s -allow-unregistered-dialect \
// RUN:     --test-transform-dialect-interpreter -verify-diagnostics \
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

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1
  %1 = transform.structured.match ops{["test.some_op"]} in %arg1
  transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [20]
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

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["affine.min"]} in %arg1
  %1 = transform.structured.match ops{["test.workgroup_id"]} in %arg1
  %2 = transform.structured.match ops{["test.thread_id"]} in %arg1
  transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 0] and [31, 31]
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["affine.min"]} in %arg1
  // expected-error@+1 {{incorrect number of lower bounds, expected 0 but found 1}}
  transform.affine.simplify_bounded_affine_ops %0 with [] within [0] and []
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["affine.min"]} in %arg1
  // expected-error@+1 {{incorrect number of upper bounds, expected 0 but found 1}}
  transform.affine.simplify_bounded_affine_ops %0 with [] within [] and [5]
}

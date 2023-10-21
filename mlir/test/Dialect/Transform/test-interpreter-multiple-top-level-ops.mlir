// RUN: mlir-opt %s --test-transform-dialect-interpreter='enforce-single-top-level-transform-op=0' -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

transform.sequence failures(propagate) {
// CHECK: transform.sequence
^bb0(%arg0: !transform.any_op):
}

transform.sequence failures(propagate) {
// CHECK: transform.sequence
^bb0(%arg0: !transform.any_op):
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %match = transform.structured.match ops{["transform.get_parent_op"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.test_print_remark_at_operand %match, "found get_parent_op" : !transform.any_op
}

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %op = transform.structured.match ops{[]} in %arg0 : (!transform.any_op) -> !transform.any_op
  // expected-remark @below{{found get_parent_op}}
  %1 = transform.get_parent_op %op : (!transform.any_op) -> !transform.any_op
}

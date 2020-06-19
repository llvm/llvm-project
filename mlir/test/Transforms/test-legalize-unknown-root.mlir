// RUN: mlir-opt %s -test-legalize-unknown-root-patterns | FileCheck %s

// Test that all `test` dialect operations are removed.
// CHECK-LABEL: func @remove_all_ops
func @remove_all_ops(%arg0: i32) {
  // CHECK-NEXT: return
  %0 = "test.illegal_op_a"() : () -> i32
  %1 = "test.illegal_op_b"() : () -> i32
  %2 = "test.illegal_op_c"() : () -> i32
  %3 = "test.illegal_op_d"() : () -> i32
  %4 = "test.illegal_op_e"() : () -> i32
  return
}

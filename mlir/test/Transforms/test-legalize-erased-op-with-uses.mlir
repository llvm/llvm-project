// RUN: mlir-opt %s -test-legalize-unknown-root-patterns -verify-diagnostics

// Test that an error is emitted when an operation is marked as "erased", but
// has users that live across the conversion.
func.func @remove_all_ops(%arg0: i32) -> i32 {
  // expected-error@below {{failed to legalize unresolved materialization from () to ('i32') that remained live after conversion}}
  %0 = "test.illegal_op_a"() : () -> i32
  // expected-note@below {{see existing live user here}}
  return %0 : i32
}

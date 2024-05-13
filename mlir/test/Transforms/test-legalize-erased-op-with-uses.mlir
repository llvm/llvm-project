// RUN: mlir-opt %s -test-legalize-unknown-root-patterns -verify-diagnostics

// Test that an error is emitted when an operation is marked as "erased", but
// has users that live across the conversion.
func.func @remove_all_ops(%arg0: i32) -> i32 {
  // expected-error@below {{failed to legalize operation 'test.illegal_op_a' marked as erased}}
  %0 = "test.illegal_op_a"() : () -> i32
  // expected-note@below {{found live user of result #0: func.return %0 : i32}}
  return %0 : i32
}

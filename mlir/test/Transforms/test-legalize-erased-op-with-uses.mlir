// RUN: mlir-opt %s -test-legalize-unknown-root-patterns -split-input-file -verify-diagnostics | FileCheck %s

// Test that an error is emitted when an operation is marked as "erased", but
// has users that live across the conversion.
func.func @remove_all_ops(%arg0: i32) -> i32 {
  // expected-error@below {{failed to legalize unresolved materialization from () to ('i32') that remained live after conversion}}
  %0 = "test.illegal_op_a"() : () -> i32
  // expected-note@below {{see existing live user here}}
  return %0 : i32
}

// -----

// Test that folding an identity cast does not hide a live use of an explicitly
// erased result.
func.func @remove_op_through_folded_identity_cast() -> i32 {
  %0 = "test.illegal_op_a"() : () -> i32
  // expected-error@below {{failed to legalize unresolved materialization from () to ('i32') that remained live after conversion}}
  %1 = builtin.unrealized_conversion_cast %0 : i32 to i32
  // expected-note@below {{see existing live user here}}
  return %1 : i32
}

// -----

// CHECK-LABEL: func.func @compose_partial_1_to_n_erasure
// CHECK-SAME:  (%[[ARG:.*]]: i32) -> i32 {
// CHECK-NEXT:    return %[[ARG]] : i32
// CHECK-NEXT:  }
func.func @compose_partial_1_to_n_erasure(%arg0: i32) -> i32 {
  %0 = "test.illegal_op_b"() : () -> i32
  %1 = "test.cast"(%0, %arg0) : (i32, i32) -> i32
  return %1 : i32
}

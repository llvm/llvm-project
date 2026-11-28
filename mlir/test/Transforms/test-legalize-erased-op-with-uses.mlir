// RUN: mlir-opt %s -test-legalize-unknown-root-patterns -split-input-file -verify-diagnostics

// Test that an error is emitted when an operation is marked as "erased", but
// has users that live across the conversion.
func.func @remove_all_ops(%arg0: i32) -> i32 {
  // expected-error@below {{failed to legalize unresolved materialization from () to ('i32') that remained live after conversion}}
  %0 = "test.illegal_op_a"() : () -> i32
  // expected-note@below {{see existing live user here}}
  return %0 : i32
}

// -----

// Test that diagnostics for a failed conversion do not crash when printing
// malformed SPIR-V IR produced during unresolved materialization cleanup.
module {
  spirv.func @remove_test_ops() -> i32 "None" {
    %cst1_i32 = spirv.Constant 1 : i32
    // expected-error@below {{failed to legalize unresolved materialization from () to ('i32') that remained live after conversion}}
    %0 = test.with_bounds {smax = 0 : si32, smin = 0 : si32, umax = 0 : i32, umin = 0 : ui32} : i32
    %1 = builtin.unrealized_conversion_cast %cst1_i32 : i32 to i32
    // expected-note@below {{see existing live user here}}
    %2 = spirv.IAdd %0, %1 : i32
    %3 = builtin.unrealized_conversion_cast %2 : i32 to i32
    %4 = test.reflect_bounds %3 : i32
    %5 = builtin.unrealized_conversion_cast %4 : i32 to i32
    spirv.ReturnValue %5 : i32
  }
}

// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}))' -verify-diagnostics=only-expected

// Test that only expected errors are reported.
// reports {{illegal operation}} but unchecked
func.func @TestAlwaysIllegalOperationPass1() {
  return
}

// expected-error@+1 {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass2() {
  return
}

// reports {{illegal operation}} but unchecked
func.func @TestAlwaysIllegalOperationPass3() {
  return
}

// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}))' -verify-diagnostics

// Test that all errors are reported.
// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass1() {
  return
}

// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass2() {
  return
}

// expected-error@below {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass3() {
  return
}

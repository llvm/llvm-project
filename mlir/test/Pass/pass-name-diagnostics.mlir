// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-pass-failure{gen-diagnostics}))' -emit-pass-error-on-failure -verify-diagnostics=only-expected

// expected-error@+2 {{failed to run pass}}
// expected-error@+1 {{illegal operation}}
func.func @TestAlwaysIllegalOperationPass1() {
  return
}

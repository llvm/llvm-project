// RUN: mlir-opt %s -verify-diagnostics -pass-pipeline='builtin.module(func.func(test-interface-pass))' -o /dev/null

// Test that we run the interface pass on the function.

// expected-remark@below {{Executing interface pass on operation}}
func.func @main() {
  return
}

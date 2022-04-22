// RUN: mlir-opt %s -pass-pipeline='func.func(test-pass-invalid-parent)' -verify-diagnostics

// Test that we properly report errors when the parent becomes invalid after running a pass
// on a child operation.
// expected-error@below {{'some_unknown_func' does not reference a valid function}}
func.func @TestCreateInvalidCallInPass() {
  return
}

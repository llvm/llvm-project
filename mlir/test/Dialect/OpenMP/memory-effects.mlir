// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

// Check the trait on omp.masked itself, independently of its body.
func.func @masked_recursive_memory_effects() {
  // expected-remark@+1 {{recursive memory effects}}
  omp.masked {
    omp.terminator
  } {test_recursive_memory_effects}
  return
}
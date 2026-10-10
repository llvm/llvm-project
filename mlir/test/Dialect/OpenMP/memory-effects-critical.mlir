// RUN: mlir-opt %s --test-side-effects --verify-diagnostics

func.func @critical_effects() {
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  omp.critical {
    omp.terminator
  }
  return
}

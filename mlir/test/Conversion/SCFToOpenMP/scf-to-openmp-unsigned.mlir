// RUN: mlir-opt -convert-scf-to-openmp='num-threads=4' -verify-diagnostics %s

// The OpenMP loop bounds are signed; an unsigned scf.parallel is left
// unconverted, which the pass reports.
func.func @parallel_unsigned(%arg0: index, %arg1: index, %arg2: index) {
  // expected-error @+1 {{unconverted operation found}}
  scf.parallel unsigned (%i) = (%arg0) to (%arg1) step (%arg2) {
    "test.payload"(%i) : (index) -> ()
  }
  return
}

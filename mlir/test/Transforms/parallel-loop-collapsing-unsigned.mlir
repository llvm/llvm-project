// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}))' -verify-diagnostics

// The normalization performed by the collapsing assumes signed bounds.
func.func @collapse_unsigned(%lb0: index, %ub0: index, %lb1: index, %ub1: index) {
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{'scf.parallel' op with unsigned bounds is not supported}}
  scf.parallel unsigned (%i0, %i1) = (%lb0, %lb1) to (%ub0, %ub1) step (%c1, %c1) {
    %result = "magic.op"(%i0, %i1) : (index, index) -> index
  }
  return
}

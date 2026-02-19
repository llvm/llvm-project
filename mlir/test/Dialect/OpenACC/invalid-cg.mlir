// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// -----

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c4 = arith.constant 4 : index
scf.parallel (%iv) = (%c0) to (%c4) step (%c1) {
  scf.reduce
// expected-error@+1 {{expected one of ::mlir::gpu::Processor enum names}}
} {acc.par_dims = #acc<par_dims[thread_x, bad_dim]>}

// -----

%c0_2 = arith.constant 0 : index
%c1_2 = arith.constant 1 : index
%c4_2 = arith.constant 4 : index
scf.parallel (%iv) = (%c0_2) to (%c4_2) step (%c1_2) {
  scf.reduce
// expected-error@+1 {{expected one of ::mlir::gpu::Processor enum names}}
} {acc.par_dims = #acc<par_dims[gang]>}

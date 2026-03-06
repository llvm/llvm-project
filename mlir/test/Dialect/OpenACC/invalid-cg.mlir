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

// -----

// expected-note@+1 {{prior use here}}
%c32 = arith.constant 32 : index
// expected-error@+1 {{use of value '%c32' expects different type than prior uses: '!acc.par_width' vs 'index'}}
acc.compute_region launch(%arg0 = %c32) {
  acc.yield
} {origin = "acc.parallel"}

// -----

// Use generic form to introduce an extra block argument.
%c64 = arith.constant 64 : index
%w = acc.par_width %c64 {par_dim = #acc.par_dim<thread_x>}
// expected-error@+1 {{'acc.compute_region' op expected 1 block arguments (launch + input), got 2}}
"acc.compute_region"(%w) <{operandSegmentSizes = array<i32: 1, 0, 0>}> ({
^bb0(%arg0: index, %extra: index):
  "acc.yield"() : () -> ()
}) {origin = "acc.parallel"} : (!acc.par_width) -> ()

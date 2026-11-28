// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" -verify-diagnostics

// A row wider than a subgroup is synchronized with an NVVM named barrier,
// which has no portable equivalent. The pass runs here with the default
// device type rather than nvidia, so the per-row barrier is reported as not
// yet implemented instead of being emitted.

func.func @per_row_barrier_non_nvidia(%a: memref<128xf32>, %b: memref<128xf32>) {
  %c1_pw = arith.constant 1 : index
  %c128_pw = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128_pw par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%ma = %a, %mb = %b) : (memref<128xf32>, memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %one = arith.constant 1.0 : f32
    scf.parallel (%g) = (%c0) to (%kbx) step (%c1) {
      scf.for %it = %c0 to %c4 step %c1 {
        // expected-error @below {{not yet implemented: per-row barrier to support worker parallelism on non-NVIDIA device}}
        scf.parallel (%i) = (%c0) to (%c128) step (%c1) {
          memref.store %one, %ma[%i] : memref<128xf32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        %v = memref.load %ma[%c3] : memref<128xf32>
        memref.store %v, %mb[%it] : memref<128xf32>
      }
      scf.reduce
    } {acc.par_dims = #acc<par_dims[block_x]>}
    acc.yield
  } <{origin = "acc.kernels"}>
  return
}

// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" \
// RUN:   -verify-diagnostics

func.func @mixed_scope_worker_reduction_combine(
    %other: memref<i32>, %result: memref<i32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block_y = acc.par_width %c1 {par_dim = #acc.par_dim<block_y>}
  %thread_y = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %thread_x = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>]
        : () -> !acc.private_type<memref<i32>>
    // expected-error@+1 {{failed to legalize operation 'acc.compute_region' that was explicitly marked illegal}}
    acc.compute_region launch(%by = %block_y, %ty = %thread_y, %tx = %thread_x)
        ins(%private_arg = %private, %other_arg = %other,
            %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c0_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        acc.predicate_region {
          acc.reduction_combine %local into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
          // expected-error@+1 {{mixed worker-private and non-worker-private reduction combines require incompatible ThreadY predication}}
          acc.reduction_combine %other_arg into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

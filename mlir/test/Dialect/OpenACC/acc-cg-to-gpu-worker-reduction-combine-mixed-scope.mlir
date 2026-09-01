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
          // expected-error@+1 {{operations in the same predicate region require incompatible ThreadY predication}}
          acc.reduction_combine %other_arg into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_x]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

func.func @worker_combine_with_single_store(%result: memref<i32>) {
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
        ins(%private_arg = %private, %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c7_i32 = arith.constant 7 : i32
      %false = arith.constant false
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        %selected = arith.select %false, %local, %result_arg : memref<i32>
        scf.parallel (%worker_iv) = (%c0) to (%ty) step (%c1_inner) {
          memref.store %c7_i32, %local[] : memref<i32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        acc.predicate_region {
          // expected-error@+1 {{operations in the same predicate region require incompatible ThreadY predication}}
          memref.store %c7_i32, %selected[] : memref<i32>
          acc.reduction_combine %local into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

func.func @worker_combine_with_atomic_update(%result: memref<i32>) {
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
        ins(%private_arg = %private, %result_arg = %result)
        : (!acc.private_type<memref<i32>>, memref<i32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      scf.parallel (%block_iv) = (%c0) to (%by) step (%c1_inner) {
        %local = acc.private_local %private_arg
            : (!acc.private_type<memref<i32>>) -> memref<i32>
        acc.predicate_region {
          // expected-error@+1 {{operations in the same predicate region require incompatible ThreadY predication}}
          acc.atomic.update %result_arg : memref<i32> {
          ^bb0(%current: i32):
            %next = arith.addi %current, %c1_i32 : i32
            acc.yield %next : i32
          }
          acc.reduction_combine %local into %result_arg <add> : memref<i32>
              {acc.par_dims = #acc<par_dims[block_y, thread_y]>}
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_y]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

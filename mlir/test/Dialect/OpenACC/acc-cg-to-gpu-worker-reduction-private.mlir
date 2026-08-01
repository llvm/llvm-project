// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// CHECK-LABEL: func.func @worker_reduction_private
// CHECK: acc.gpu_shared_memory
// CHECK-SAME: num_copies = 4

func.func @worker_reduction_private() {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %block = acc.par_width %c2 {par_dim = #acc.par_dim<block_x>}
  %worker = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %vector = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %private = acc.privatize [#acc<par_dims[thread_y]>] : () -> !acc.private_type<memref<i32>>
    acc.compute_region launch(%bx = %block, %wy = %worker, %vx = %vector)
        ins(%private_arg = %private) : (!acc.private_type<memref<i32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      scf.parallel (%block_iv) = (%c0) to (%bx) step (%c1) {
        scf.parallel (%k) = (%block_iv) to (%bx) step (%bx) {
          %local = acc.private_local %private_arg
              {acc.par_dims = #acc<par_dims[thread_y]>}
              : (!acc.private_type<memref<i32>>) -> memref<i32>
          memref.store %c0_i32, %local[] : memref<i32>
          scf.parallel (%worker_iv) = (%c0) to (%wy) step (%c1) {
            scf.reduce
          } {acc.par_dims = #acc<par_dims[thread_y]>}
          %value = memref.load %local[] : memref<i32>
          acc.reduction_accumulate %value to %local <add>
              : i32 -> memref<i32> {par_dims = #acc<par_dims[thread_y]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

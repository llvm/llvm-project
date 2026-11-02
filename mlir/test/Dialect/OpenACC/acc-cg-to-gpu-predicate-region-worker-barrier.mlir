// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// For a predicate region in a worker loop, generate a barrier to reconverge
// all threads within the worker.

// CHECK-LABEL: func.func @predicate_region_in_worker_loop
// CHECK:       gpu.launch
// CHECK:         scf.parallel
// CHECK:           scf.if
// CHECK:             memref.store
// CHECK:           gpu.barrier scope <subgroup>
// CHECK-NOT:       gpu.barrier{{$}}
func.func @predicate_region_in_worker_loop(%arg0: memref<4xi32>) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %par_bx = acc.par_width %c1 par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty,
                              %vector = %par_tx)
        ins(%out = %arg0) : (memref<4xi32>) {
      %c0 = arith.constant 0 : index
      %c1_inner = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      scf.parallel (%worker_iv) = (%c0) to (%worker) step (%c1_inner) {
        %ub = arith.addi %worker_iv, %c1_inner : index
        scf.parallel (%seq_iv) = (%c0) to (%ub) step (%c1_inner) {
          acc.predicate_region {
            memref.store %c1_i32, %out[%worker_iv] : memref<4xi32>
          }
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_y]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

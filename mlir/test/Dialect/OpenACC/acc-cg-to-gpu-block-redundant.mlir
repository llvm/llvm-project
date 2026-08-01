// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A loop parallelized at the vector (thread_x) level and marked
// `acc.gpu_block_redundant` runs redundantly on every thread block. Because the
// block dimension is redundant (all blocks execute) and thread_x is the active
// worksharing dimension, there is no inactive parallel dimension left to
// serialize on, so a predicate_region inside the loop must lower without any
// blockIdx/threadIdx predication (no scf.if / arith.cmpi guard on the store).

// CHECK-LABEL: func.func @block_redundant_vector_no_predicate
// CHECK:         gpu.launch
// CHECK-NOT:     acc.predicate_region
// CHECK-NOT:     scf.if
// CHECK-NOT:     arith.cmpi
// CHECK:         memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<32xi32>

func.func @block_redundant_vector_no_predicate(%arg0: memref<32xi32>) {
  %c32 = arith.constant 32 : index
  %par_bx = acc.par_width %c32 {par_dim = #acc.par_dim<block_x>}
  %par_tx = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    acc.compute_region launch(%grid = %par_bx, %block = %par_tx)
        ins(%arg10 = %arg0) : (memref<32xi32>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32_0 = arith.constant 32 : index
      scf.parallel (%i) = (%c0) to (%c32_0) step (%c1) {
        acc.predicate_region {
          %c42 = arith.constant 42 : i32
          memref.store %c42, %arg10[%i] : memref<32xi32>
        }
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>, acc.gpu_block_redundant = #acc.gpu_block_redundant}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A scalar bridged out of an acc.predicate_region is stored to gang-shared
// memory by the predicated thread and read by all threads after a barrier.
// When the predicate region sits in a gang/block-level sequential loop the
// shared slot is reused every iteration, so a reuse barrier must also be
// emitted BEFORE the predicated store: otherwise the next iteration's store
// can clobber the slot before all threads have read the current value (WAR).

// Inside a block-level sequential loop: barrier BEFORE and AFTER the store.
// CHECK-LABEL: func.func @reuse_barrier_in_block_seq_loop
// CHECK:       scf.parallel
// CHECK:         gpu.barrier
// CHECK-NEXT:    scf.if
// CHECK:         gpu.barrier
// CHECK:         memref.load
func.func @reuse_barrier_in_block_seq_loop() {
  %c256_pw = arith.constant 256 : index
  %c1024_pw = arith.constant 1024 : index
  %par_bx = acc.par_width %c256_pw {par_dim = #acc.par_dim<block_x>}
  %par_tx = acc.par_width %c1024_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv0 = acc.privatize : () -> !acc.private_type<memref<i64>>
    acc.compute_region launch(%grid = %par_bx, %block = %par_tx) ins(%arg10 = %priv0) : (!acc.private_type<memref<i64>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      scf.parallel (%gang_iv) = (%c0) to (%grid) step (%c1) {
        scf.parallel (%seq_iv) = (%c0) to (%c256) step (%c1) {
          %pl0 = acc.private_local %arg10 : (!acc.private_type<memref<i64>>) -> memref<i64>
          acc.predicate_region {
            %val = arith.index_cast %seq_iv : index to i64
            memref.store %val, %pl0[] : memref<i64>
          }
          %v0 = memref.load %pl0[] : memref<i64>
          scf.parallel (%vec_iv) = (%c0) to (%block) step (%c1) {
            memref.store %v0, %pl0[] : memref<i64>
            scf.reduce
          } {acc.par_dims = #acc<par_dims[thread_x]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[sequential]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// Not inside a sequential loop: only the post-store barrier, no pre-store one.
// CHECK-LABEL: func.func @no_reuse_barrier_outside_seq_loop
// CHECK-NOT:   gpu.barrier
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK:       memref.load
func.func @no_reuse_barrier_outside_seq_loop() {
  %c256_pw = arith.constant 256 : index
  %c1024_pw = arith.constant 1024 : index
  %par_bx = acc.par_width %c256_pw {par_dim = #acc.par_dim<block_x>}
  %par_tx = acc.par_width %c1024_pw {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    %priv0 = acc.privatize : () -> !acc.private_type<memref<i64>>
    acc.compute_region launch(%grid = %par_bx, %block = %par_tx) ins(%arg10 = %priv0) : (!acc.private_type<memref<i64>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%gang_iv) = (%c0) to (%grid) step (%c1) {
        %pl0 = acc.private_local %arg10 : (!acc.private_type<memref<i64>>) -> memref<i64>
        acc.predicate_region {
          %val = arith.index_cast %gang_iv : index to i64
          memref.store %val, %pl0[] : memref<i64>
        }
        %v0 = memref.load %pl0[] : memref<i64>
        scf.parallel (%vec_iv) = (%c0) to (%block) step (%c1) {
          memref.store %v0, %pl0[] : memref<i64>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

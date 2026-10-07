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
  %par_bx = acc.par_width %c256_pw par_dim(#acc.par_dim<block_x>)
  %par_tx = acc.par_width %c1024_pw par_dim(#acc.par_dim<thread_x>)
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
    } <{origin = "acc.parallel"}>
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
  %par_bx = acc.par_width %c256_pw par_dim(#acc.par_dim<block_x>)
  %par_tx = acc.par_width %c1024_pw par_dim(#acc.par_dim<thread_x>)
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
    } <{origin = "acc.parallel"}>
  }
  return
}

// Predicate stores to both a gang-private slot (never reused in parallel) and a
// worker-private slot (reused by a nested vector loop). The gang store must
// not hide the worker WAR: emit a per-row pre-store barrier, not a workgroup
// one and not no barrier at all.
// CHECK-LABEL: func.func @mixed_gang_store_worker_reuse
// CHECK:       gpu.launch
// CHECK:         scf.parallel
// CHECK:           gpu.barrier scope <subgroup>
// CHECK-NEXT:      scf.if
// CHECK:           gpu.barrier scope <subgroup>
func.func @mixed_gang_store_worker_reuse() {
  %c256_pw = arith.constant 256 : index
  %c4_pw = arith.constant 4 : index
  %c32_pw = arith.constant 32 : index
  %par_bx = acc.par_width %c256_pw par_dim(#acc.par_dim<block_x>)
  %par_ty = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
  %par_tx = acc.par_width %c32_pw par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    %gang_priv = acc.privatize par_dims(#acc<par_dims[block_x]>) : () -> !acc.private_type<memref<i64>>
    %worker_priv = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<i64>>
    acc.compute_region launch(%grid = %par_bx, %worker = %par_ty, %block = %par_tx)
        ins(%arg_gang = %gang_priv, %arg_worker = %worker_priv)
        : (!acc.private_type<memref<i64>>, !acc.private_type<memref<i64>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%gang_iv) = (%c0) to (%grid) step (%c1) {
        scf.parallel (%worker_iv) = (%c0) to (%worker) step (%c1) {
          %plg = acc.private_local %arg_gang : (!acc.private_type<memref<i64>>) -> memref<i64>
          %plw = acc.private_local %arg_worker : (!acc.private_type<memref<i64>>) -> memref<i64>
          scf.parallel (%k) = (%c0) to (%c8) step (%c1) {
            acc.predicate_region {
              %val = arith.index_cast %k : index to i64
              memref.store %val, %plg[] : memref<i64>
              memref.store %val, %plw[] : memref<i64>
            }
            scf.parallel (%vec_iv) = (%c0) to (%block) step (%c1) {
              %v = memref.load %plw[] : memref<i64>
              memref.store %v, %plw[] : memref<i64>
              scf.reduce
            } {acc.par_dims = #acc<par_dims[thread_x]>}
            scf.reduce
          } {acc.par_dims = #acc<par_dims[sequential]>}
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  return
}

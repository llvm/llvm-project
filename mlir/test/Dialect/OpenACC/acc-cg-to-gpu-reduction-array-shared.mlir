// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// A block-shared array accumulator (too large for a per-thread stack alloca)
// must NOT run a per-element gpu.all_reduce: all threads share the slot, so
// all_reduce would scale the block partial by the thread count. The block
// partial is already in place and the atomic combine merges across blocks, so
// the accumulate lowers away to nothing (like the block-only case).
// Here every thread updates element 0, so the in-place update must be atomic
// or the threads race and partials are lost.

// CHECK-LABEL: func.func @array_reduction_shared
// CHECK: gpu.launch
// CHECK: memref.atomic_rmw addi
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array
// CHECK-NOT: acc.bounds

func.func @array_reduction_shared(%arg0: memref<8192xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<8192xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<8192xi32>
  acc.kernel_environment dataOperands(%0 : memref<8192xi32>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<8192xi32>) {
      %c8192 = arith.constant 8192 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %2 = acc.reduction_init %arg2 <add> : memref<8192xi32> {
        %alloc = memref.alloc() : memref<8192xi32>
        scf.parallel (%i) = (%c0) to (%c8192) step (%c1) {
          memref.store %c0_i32, %alloc[%i] : memref<8192xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloc : memref<8192xi32>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%ktx) step (%c1) {
          %3 = memref.load %2[%c0] : memref<8192xi32>
          %4 = arith.addi %3, %c1_i32 : i32
          memref.store %4, %2[%c0] : memref<8192xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c8192 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> : memref<8192xi32> <{par_dims = #acc<par_dims[block_x, thread_x]>}>
      acc.reduction_combine_region %2 into %arg2 : memref<8192xi32> {
        scf.for %i = %c0 to %c8192 step %c1 {
          %3 = memref.load %2[%i] : memref<8192xi32>
          %4 = memref.load %arg2[%i] : memref<8192xi32>
          %5 = arith.addi %3, %4 : i32
          memref.store %5, %arg2[%i] : memref<8192xi32>
        }
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%0 : memref<8192xi32>) to varPtr(%arg0 : memref<8192xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// The same block-shared accumulator, but each thread updates its own element.
// The update is already race-free, so it must stay a plain store.

// CHECK-LABEL: func.func @array_reduction_shared_partitioned
// CHECK: gpu.launch
// CHECK-NOT: memref.atomic_rmw
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array

func.func @array_reduction_shared_partitioned(%arg0: memref<8192xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<8192xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<8192xi32>
  acc.kernel_environment dataOperands(%0 : memref<8192xi32>) {
    %c1_pw = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
    acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%arg2 = %0) : (memref<8192xi32>) {
      %c8192 = arith.constant 8192 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %2 = acc.reduction_init %arg2 <add> : memref<8192xi32> {
        %alloc = memref.alloc() : memref<8192xi32>
        scf.parallel (%i) = (%c0) to (%c8192) step (%c1) {
          memref.store %c0_i32, %alloc[%i] : memref<8192xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        acc.yield %alloc : memref<8192xi32>
      }
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        scf.parallel (%tx_iv) = (%c0) to (%c8192) step (%c1) {
          %3 = memref.load %2[%tx_iv] : memref<8192xi32>
          %4 = arith.addi %3, %c1_i32 : i32
          memref.store %4, %2[%tx_iv] : memref<8192xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      %b = acc.bounds extent(%c8192 : index)
      acc.reduction_accumulate_array %2 bounds(%b) <add> : memref<8192xi32> <{par_dims = #acc<par_dims[block_x, thread_x]>}>
      acc.reduction_combine_region %2 into %arg2 : memref<8192xi32> {
        scf.for %i = %c0 to %c8192 step %c1 {
          %3 = memref.load %2[%i] : memref<8192xi32>
          %4 = memref.load %arg2[%i] : memref<8192xi32>
          %5 = arith.addi %3, %4 : i32
          memref.store %5, %arg2[%i] : memref<8192xi32>
        }
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%0 : memref<8192xi32>) to varPtr(%arg0 : memref<8192xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// A gang-scoped array private_local (block-only storage dims) feeding a
// worker-level accumulate must not take the per-thread all_reduce path. Every
// worker updates element 0 of the gang-private slot, so the update must be
// atomic.

// CHECK-LABEL: func.func @array_reduction_gang_storage_thread_accum
// CHECK: gpu.launch
// CHECK: memref.atomic_rmw addi
// CHECK-NOT: gpu.all_reduce
// CHECK-NOT: acc.reduction_accumulate_array

func.func @array_reduction_gang_storage_thread_accum(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %bx = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4 par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c32 par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize [#acc<par_dims[block_x]>] : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c4_idx = arith.constant 4 : index
      scf.parallel (%bx_iv) = (%c0) to (%kbx) step (%c1) {
        %local = acc.private_local %priv {acc.par_dims = #acc<par_dims[block_x]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
        scf.for %i = %c0 to %c4_idx step %c1 {
          memref.store %c0_i32, %local[%i] : memref<4xi32>
        }
        scf.parallel (%wy_iv) = (%c0) to (%kwy) step (%c1) {
          %3 = memref.load %local[%c0] : memref<4xi32>
          %4 = arith.addi %3, %c1_i32 : i32
          memref.store %4, %local[%c0] : memref<4xi32>
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_y]>}
        %b = acc.bounds extent(%c4_idx : index)
        acc.reduction_accumulate_array %local bounds(%b) <add> : memref<4xi32> <{par_dims = #acc<par_dims[thread_y]>}>
        scf.reduce
      } {acc.par_dims = #acc<par_dims[block_x]>}
      acc.yield
    } {origin = "acc.parallel"}
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

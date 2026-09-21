// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu{device-type=nvidia}))" --split-input-file | FileCheck %s

// A privatization reduced across thread_y but materialized once per block is
// shared by every worker row, so the whole workgroup must reconverge.

// CHECK-LABEL: func.func @worker_shared_private
// CHECK:       gpu.launch
// CHECK:       scf.if
// CHECK:       gpu.barrier
// CHECK-NOT:   gpu.barrier scope <subgroup>
// CHECK-NOT:   nvvm.barrier

func.func @worker_shared_private(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      acc.predicate_region {
        scf.for %i = %c0 to %c4 step %c1 {
          memref.store %c0_i32, %local[%i] : memref<4xi32>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

// -----

// thread_y active: every row owns its slot, so the cheaper per-row barrier is
// kept, in the non-aligned form since only one row reaches it.

// CHECK-LABEL: func.func @worker_private_per_row
// CHECK:       gpu.launch
// CHECK:       scf.if
// CHECK:       nvvm.barrier id = %{{.*}} number_of_threads = %{{.*}} aligned = false

func.func @worker_private_per_row(%arg0: memref<4xi32>) {
  %0 = acc.copyin varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r") -> memref<4xi32>
  acc.kernel_environment dataOperands(%0 : memref<4xi32>) {
    %c1_pw = arith.constant 1 : index
    %c4_pw = arith.constant 4 : index
    %c64_pw = arith.constant 64 : index
    %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
    %wy = acc.par_width %c4_pw par_dim(#acc.par_dim<thread_y>)
    %tx = acc.par_width %c64_pw par_dim(#acc.par_dim<thread_x>)
    %private = acc.privatize par_dims(#acc<par_dims[block_x, thread_y]>) : () -> !acc.private_type<memref<4xi32>>
    acc.compute_region launch(%kbx = %bx, %kwy = %wy, %ktx = %tx) ins(%arg2 = %0, %priv = %private) : (memref<4xi32>, !acc.private_type<memref<4xi32>>) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0_i32 = arith.constant 0 : i32
      %local = acc.private_local %priv {acc.active_par_dims = #acc<active_par_dims[block_x, thread_y]>, acc.par_dims = #acc<par_dims[block_x, thread_y]>} : (!acc.private_type<memref<4xi32>>) -> memref<4xi32>
      acc.predicate_region {
        scf.for %i = %c0 to %c4 step %c1 {
          memref.store %c0_i32, %local[%i] : memref<4xi32>
        }
      }
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%0 : memref<4xi32>) to varPtr(%arg0 : memref<4xi32>) dataClause(acc_reduction) implicit(true) name("r")
  return
}

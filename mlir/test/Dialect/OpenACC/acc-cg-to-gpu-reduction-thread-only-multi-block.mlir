// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// A reduction partitioned only across threads still leaves one partial per
// block when the launch spans several blocks, and every block combines its
// partial into the same shared slot. The combine must therefore be atomic even
// though no block dimension appears in the reduction's parallel dims.

// CHECK-LABEL: func.func @thread_only_combine_multi_block
// CHECK: acc.atomic.update

func.func @thread_only_combine_multi_block(%arg_slot: memref<i32>, %arg_res: memref<i32>) {
  %c8_pw = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c8_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_slot = %arg_slot, %a_res = %arg_res) : (memref<i32>, memref<i32>) {
    acc.predicate_region {
      acc.reduction_combine %a_slot into %a_res <add> par_dims(#acc<par_dims[thread_x]>) : memref<i32>
    }
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

// -----

// A single-block launch has only one partial, so the same thread-only combine
// needs no atomic and stays a plain load/combine/store.

// CHECK-LABEL: func.func @thread_only_combine_single_block
// CHECK-NOT: acc.atomic.update
// CHECK: memref.store

func.func @thread_only_combine_single_block(%arg_slot: memref<i32>, %arg_res: memref<i32>) {
  %c1_pw = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %bx = acc.par_width %c1_pw par_dim(#acc.par_dim<block_x>)
  %tx = acc.par_width %c128 par_dim(#acc.par_dim<thread_x>)
  acc.compute_region launch(%kbx = %bx, %ktx = %tx) ins(%a_slot = %arg_slot, %a_res = %arg_res) : (memref<i32>, memref<i32>) {
    acc.predicate_region {
      acc.reduction_combine %a_slot into %a_res <add> par_dims(#acc<par_dims[thread_x]>) : memref<i32>
    }
    acc.yield
  } <{origin = "acc.parallel"}>
  return
}

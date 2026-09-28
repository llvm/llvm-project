// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" --split-input-file | FileCheck %s

// Without the attribute the update runs on every block: only the thread dim
// is predicated.
// CHECK-LABEL: func.func @gang_redundant_update
// CHECK:       gpu.launch blocks(%[[BID:[a-z0-9_]+]], %{{.*}}) in ({{.*}}) threads(%[[TID:[a-z0-9_]+]], %{{.*}}) in ({{.*}}) {
// CHECK:       %[[P:.*]] = arith.cmpi eq, %[[TID]], %c0
// CHECK-NOT:   arith.cmpi eq, %[[BID]]
// CHECK:       scf.if %[[P]]

func.func @gang_redundant_update(%arg0: memref<f32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
  %1 = acc.par_width %c2 par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    acc.compute_region launch(%arg1 = %0, %arg2 = %1) ins(%arg10 = %arg0) : (memref<f32>) {
      %cst = arith.constant 1.000000e+00 : f32
      %2 = memref.load %arg10[] : memref<f32>
      %3 = arith.addf %2, %cst : f32
      acc.predicate_region {
        memref.store %3, %arg10[] : memref<f32>
      }
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// -----

// An empty active set means no dim runs it unpredicated, so the block dim is
// predicated too and one block applies the update.
// CHECK-LABEL: func.func @single_block_update
// CHECK:       gpu.launch blocks(%[[BID:[a-z0-9_]+]], %{{.*}}) in ({{.*}}) threads(%[[TID:[a-z0-9_]+]], %{{.*}}) in ({{.*}}) {
// CHECK:       %[[PB:.*]] = arith.cmpi eq, %[[BID]], %c0
// CHECK:       %[[PT:.*]] = arith.cmpi eq, %[[TID]], %c0
// CHECK:       %[[P:.*]] = arith.andi %[[PT]], %[[PB]]
// CHECK:       scf.if %[[P]]

func.func @single_block_update(%arg0: memref<f32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = acc.par_width %c2 par_dim(#acc.par_dim<block_x>)
  %1 = acc.par_width %c2 par_dim(#acc.par_dim<thread_x>)
  acc.kernel_environment {
    acc.compute_region launch(%arg1 = %0, %arg2 = %1) ins(%arg10 = %arg0) : (memref<f32>) {
      %cst = arith.constant 1.000000e+00 : f32
      %2 = memref.load %arg10[] : memref<f32>
      %3 = arith.addf %2, %cst : f32
      acc.predicate_region {
        memref.store %3, %arg10[] : memref<f32>
      } {acc.active_par_dims = #acc<active_par_dims[]>}
      acc.yield
    } <{origin = "acc.kernels"}>
  }
  return
}

// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// Test that stores to worker-indexed private_local at top level preserve
// redundant gang execution. The store should NOT be guarded by block_id.x
// predicate - all gangs should execute the store.
//
// With the bug: arith.andi combines block_id.x==0 and thread_id.x==0
// With the fix: only thread_id.x==0 check (no arith.andi needed)

// CHECK-LABEL: func.func @gang_redundant_worker_private
// CHECK:       gpu.launch
// CHECK-NOT:   arith.andi
// CHECK:       memref.store

func.func @gang_redundant_worker_private() {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = acc.par_width %c4 {par_dim = #acc.par_dim<block_x>}
  %1 = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %2 = acc.par_width %c1 {par_dim = #acc.par_dim<thread_x>}
  %3 = acc.privatize [#acc<par_dims[thread_y]>] : () -> !acc.private_type<memref<1xi32>>
  acc.kernel_environment {
    acc.compute_region launch(%arg0 = %0, %arg1 = %1, %arg2 = %2) ins(%arg10 = %3) : (!acc.private_type<memref<1xi32>>) {
      %loc = acc.private_local %arg10 : (!acc.private_type<memref<1xi32>>) -> memref<1xi32>
      // Top-level predicate_region storing into the worker-private buffer.
      // This should execute redundantly across gangs (no block_id.x predicate).
      acc.predicate_region {
        %c0 = arith.constant 0 : index
        %c42 = arith.constant 42 : i32
        memref.store %c42, %loc[%c0] : memref<1xi32>
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// CHECK-LABEL: func.func @predicate_region_reduction
// CHECK-NOT:   acc.predicate_region
// CHECK:       scf.if
// CHECK-NEXT:  memref.store %c0_i32, %alloca[] : memref<i32>

func.func @predicate_region_reduction(%arg0: memref<i32>) {
  %c1 = arith.constant 1 : index
  %0 = acc.par_width %c1 {par_dim = #acc.par_dim<block_x>}
  %1 = acc.par_width %c1 {par_dim = #acc.par_dim<thread_y>}
  %2 = acc.par_width %c1 {par_dim = #acc.par_dim<thread_x>}
  acc.kernel_environment {
    acc.compute_region launch(%arg1 = %0, %arg2 = %1, %arg3 = %2) ins(%arg10 = %arg0) : (memref<i32>) {
      %c0_i32 = arith.constant 0 : i32
      %4 = acc.reduction_init %arg10 <add> : memref<i32> {
        %alloca = memref.alloca() : memref<i32>
        acc.predicate_region {
          memref.store %c0_i32, %alloca[] : memref<i32>
        }
        acc.yield %alloca : memref<i32>
      }
      acc.yield
    } {origin = "acc.parallel"}
  }
  return
}

// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s

func @omp_barrier() -> () {
  // CHECK: omp.barrier
  omp.barrier
  return
}

func @omp_taskwait() -> () {
  // CHECK: omp.taskwait
  omp.taskwait
  return
}

func @omp_taskyield() -> () {
  // CHECK: omp.taskyield
  omp.taskyield
  return
}

// CHECK-LABEL: func @omp_flush
// CHECK-SAME: %[[ARG0:.*]]: !llvm.i32
func @omp_flush(%arg0 : !llvm.i32) -> () {
  // Test without data var
  // CHECK: omp.flush
  omp.flush

  // Test with one data var
  // CHECK: omp.flush %[[ARG0]] : !llvm.i32
  "omp.flush"(%arg0) : (!llvm.i32) -> ()

  // Test with two data var
  // CHECK: omp.flush %[[ARG0]], %[[ARG0]] : !llvm.i32, !llvm.i32
  "omp.flush"(%arg0, %arg0): (!llvm.i32, !llvm.i32) -> ()

  return
}

// RUN: mlir-opt %s -pass-pipeline="func.func(test-print-topological-sort)" 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : region
//       CHECK: arith.addi {{.*}} : index
//  CHECK-NEXT: scf.for
//       CHECK: } {__test_sort_original_idx__ = 2 : i64}
//  CHECK-NEXT: arith.addi {{.*}} : i32
//  CHECK-NEXT: arith.subi {{.*}} : i32
func @region(
  %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index,
  %arg4 : i32, %arg5 : i32, %arg6 : i32,
  %buffer : memref<i32>) {
  %0 = arith.addi %arg4, %arg5 {__test_sort_original_idx__ = 0} : i32
  %idx = arith.addi %arg0, %arg1 {__test_sort_original_idx__ = 3} : index
  scf.for %arg7 = %idx to %arg2 step %arg3  {
    %2 = arith.addi %0, %arg5 : i32
    %3 = arith.subi %2, %arg6 {__test_sort_original_idx__ = 1} : i32
    memref.store %3, %buffer[] : memref<i32>
  } {__test_sort_original_idx__ = 2}
  return
}

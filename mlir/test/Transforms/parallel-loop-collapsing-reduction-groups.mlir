// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1 collapsed-indices-1=2}, canonicalize))' --mlir-print-local-scope %s | FileCheck %s

// A dimension in a group of its own is not divided back out. The reduction
// still sees every original index, and its result is preserved.

// CHECK-LABEL: func @collapse_reduction_groups(%
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK: %[[R:.*]] = scf.parallel (%[[IJ:.*]], %[[K:.*]]) = (%[[C0]], %[[C0]]) to (%[[C6]], %[[C4]]) step (%[[C1]], %[[C1]]) init (%[[ZERO]]) -> i32 {
// CHECK: %[[J:.*]] = arith.remsi %[[IJ]], %[[C3]] : index
// CHECK: %[[I:.*]] = arith.divsi %[[IJ]], %[[C3]] : index
// CHECK-NOT: arith.remsi %[[K]]
// CHECK: %[[VAL:.*]] = memref.load {{%.*}}[%[[I]], %[[J]], %[[K]]] : memref<2x3x4xi32>
// CHECK: scf.reduce(%[[VAL]] : i32) {
// CHECK: arith.addi
// CHECK: return %[[R]] : i32

func.func @collapse_reduction_groups(%m: memref<2x3x4xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %r = scf.parallel (%i, %j, %k) = (%c0, %c0, %c0) to (%c2, %c3, %c4) step (%c1, %c1, %c1) init (%zero) -> i32 {
    %v = memref.load %m[%i, %j, %k] : memref<2x3x4xi32>
    scf.reduce(%v : i32) {
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      scf.reduce.return %s : i32
    }
  }
  return %r : i32
}

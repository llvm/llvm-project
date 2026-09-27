// UNSUPPORTED: system-windows
// RUN: mlir-reduce %s -reduction-tree='traversal-mode=0 test=%S/../script/grep-select.sh' | FileCheck %s

// This test case is referenced on the website (mlir/docs/Tools/mlir-reduce.md).

func.func @func1() {
  // A func can be pruned if it's not relevant to the error.
  return
}

func.func @func2(%arg0: i1) -> f32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 2.2 : f32
  %3 = arith.constant 5.3 : f32
  %4 = arith.addi %0, %1 : i32
  %5 = arith.addf %2, %3 : f32
  %6 = arith.muli %4, %4 : i32
  %7 = arith.subi %6, %4 : i32
  %8 = arith.select %arg0, %5, %2 : f32
  %9 = arith.addf %2, %8 : f32
  return %9 : f32
}

// CHECK-NOT: func @func1
// CHECK-LABEL: func @func2
// CHECK-SAME: (%arg0: i1) -> f32
// CHECK-DAG: %[[C22:.*]] = arith.constant 2.200000e+00 : f32
// CHECK-DAG: %[[C75:.*]] = arith.constant 7.500000e+00 : f32
// CHECK: %[[SEL:.*]] = arith.select %arg0, %[[C75]], %[[C22]] : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[SEL]], %[[C22]] : f32
// CHECK: return %[[ADD]] : f32

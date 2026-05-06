// RUN: mlir-reduce %s -split-input-file -reduction-tree='traversal-mode=0 test=%S/../script/query-test.sh' | FileCheck %s
func.func @query_test(%arg0: i1) -> f32 {
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

// CHECK-LABEL: func @query_test
//  CHECK-SAME:   %[[ARG0:.*]]: i1) -> f32
//       CHECK:   %[[CONSTANT_0:.*]] = arith.constant 2.200000e+00 : f32
//       CHECK:   %[[CONSTANT_1:.*]] = arith.constant 7.500000e+00 : f32
//       CHECK:   %[[SELECT_0:.*]] = arith.select %[[ARG0]], %[[CONSTANT_1]], %[[CONSTANT_0]] : f32
//       CHECK:   %[[ADDF_0:.*]] = arith.addf %[[SELECT_0]], %[[CONSTANT_0]] : f32
//       CHECK:   return %[[ADDF_0]] : f32

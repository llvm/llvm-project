// RUN: mlir-opt %s -test-loop-zero-trip-check -split-input-file  | FileCheck %s

func.func @replace_scf_while_with_zero_trip_check(%bound : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
  %cst5 = arith.constant 5 : i32
  %res:2 = scf.while (%iter = %cst0) : (i32) -> (i32, i32) {
    %cond = arith.cmpi slt, %iter, %bound : i32
    %inv = arith.addi %bound, %cst5 : i32
    scf.condition(%cond) %iter, %inv : i32, i32
  } do {
  ^bb0(%arg1: i32, %arg2: i32):
    %next = arith.addi %arg1, %arg2 : i32
    scf.yield %next : i32
  }
  return %res#0 : i32
}

// CHECK-LABEL: func.func @replace_scf_while_with_zero_trip_check(
// CHECK-SAME:      %[[ARG0:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG:     %[[PRE_COND:.*]] = arith.cmpi slt, %[[C0]], %[[ARG0]] : i32
// CHECK-DAG:     %[[PRE_INV:.*]] = arith.addi %[[ARG0]], %[[C5]] : i32
// CHECK:         %[[IF:.*]]:2 = scf.if %[[PRE_COND]] -> (i32, i32) {
// CHECK:           %[[WHILE:.*]]:2 = scf.while (
// CHECK-SAME:          %[[ARG1:.*]] = %[[C0]], %[[ARG2:.*]] = %[[PRE_INV]]
// CHECK-SAME:      ) : (i32, i32) -> (i32, i32) {
// CHECK:             %[[NEXT:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[NEXT]], %[[ARG0]] : i32
// CHECK:             %[[INV:.*]] = arith.addi %[[ARG0]], %[[C5]] : i32
// CHECK:             scf.condition(%[[COND]]) %[[NEXT]], %[[INV]] : i32, i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32):
// CHECK:             scf.yield %[[ARG3]], %[[ARG4]] : i32, i32
// CHECK:           }
// CHECK:           scf.yield %[[WHILE]]#0, %[[WHILE]]#1 : i32, i32
// CHECK:         } else {
// CHECK:           scf.yield %[[C0]], %[[PRE_INV]] : i32, i32
// CHECK:         }
// CHECK:         return %[[IF]]#0 : i32

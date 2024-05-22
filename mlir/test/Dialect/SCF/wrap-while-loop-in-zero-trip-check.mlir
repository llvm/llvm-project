// RUN: mlir-opt %s -test-wrap-scf-while-loop-in-zero-trip-check -split-input-file  | FileCheck %s
// RUN: mlir-opt %s -test-wrap-scf-while-loop-in-zero-trip-check='force-create-check=true' -split-input-file  | FileCheck %s --check-prefix FORCE-CREATE-CHECK

func.func @wrap_while_loop_in_zero_trip_check(%bound : i32) -> i32 {
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

// CHECK-LABEL: func.func @wrap_while_loop_in_zero_trip_check(
// CHECK-SAME:      %[[BOUND:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG:     %[[PRE_COND:.*]] = arith.cmpi slt, %[[C0]], %[[BOUND]] : i32
// CHECK-DAG:     %[[PRE_INV:.*]] = arith.addi %[[BOUND]], %[[C5]] : i32
// CHECK:         %[[IF:.*]]:2 = scf.if %[[PRE_COND]] -> (i32, i32) {
// CHECK:           %[[WHILE:.*]]:2 = scf.while (
// CHECK-SAME:          %[[ARG1:.*]] = %[[C0]], %[[ARG2:.*]] = %[[PRE_INV]]
// CHECK-SAME:      ) : (i32, i32) -> (i32, i32) {
// CHECK:             %[[NEXT:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// CHECK:             %[[INV:.*]] = arith.addi %[[BOUND]], %[[C5]] : i32
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

// -----

func.func @wrap_while_loop_with_minimal_before_block(%bound : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
  %true = arith.constant true
  %cst5 = arith.constant 5 : i32
  %res = scf.while (%iter = %cst0, %arg0 = %true) : (i32, i1) -> i32 {
    scf.condition(%arg0) %iter : i32
  } do {
  ^bb0(%arg1: i32):
    %next = arith.addi %arg1, %cst5 : i32
    %cond = arith.cmpi slt, %next, %bound : i32
    scf.yield %next, %cond : i32, i1
  }
  return %res : i32
}

// CHECK-LABEL: func.func @wrap_while_loop_with_minimal_before_block(
// CHECK-SAME:      %[[BOUND:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// CHECK:         %[[IF:.*]] = scf.if %[[TRUE]] -> (i32) {
// CHECK:           %[[WHILE:.*]] = scf.while (%[[ARG1:.*]] = %[[C0]]) : (i32) -> i32 {
// CHECK:             %[[NEXT:.*]] = arith.addi %[[ARG1]], %[[C5]] : i32
// CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// CHECK:             scf.condition(%[[COND]]) %[[NEXT]] : i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[ARG2:.*]]: i32):
// CHECK:             scf.yield %[[ARG2]] : i32
// CHECK:           }
// CHECK:           scf.yield %[[WHILE]] : i32
// CHECK:         } else {
// CHECK:           scf.yield %[[C0]] : i32
// CHECK:         }
// CHECK:         return %[[IF]] : i32

// -----

func.func @wrap_do_while_loop_in_zero_trip_check(%bound : i32) -> i32 {
  %cst0 = arith.constant 0 : i32
  %cst5 = arith.constant 5 : i32
  %res = scf.while (%iter = %cst0) : (i32) -> i32 {
    %next = arith.addi %iter, %cst5 : i32
    %cond = arith.cmpi slt, %next, %bound : i32
    scf.condition(%cond) %next : i32
  } do {
  ^bb0(%arg1: i32):
    scf.yield %arg1 : i32
  }
  return %res : i32
}

// CHECK-LABEL: func.func @wrap_do_while_loop_in_zero_trip_check(
// CHECK-SAME:      %[[BOUND:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// CHECK-NOT:     scf.if
// CHECK:         %[[WHILE:.*]] = scf.while (%[[ARG1:.*]] = %[[C0]]) : (i32) -> i32 {
// CHECK:             %[[NEXT:.*]] = arith.addi %[[ARG1]], %[[C5]] : i32
// CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// CHECK:             scf.condition(%[[COND]]) %[[NEXT]] : i32
// CHECK:           } do {
// CHECK:           ^bb0(%[[ARG2:.*]]: i32):
// CHECK:             scf.yield %[[ARG2]] : i32
// CHECK:           }
// CHECK:         return %[[WHILE]] : i32

// FORCE-CREATE-CHECK-LABEL: func.func @wrap_do_while_loop_in_zero_trip_check(
// FORCE-CREATE-CHECK-SAME:      %[[BOUND:.*]]: i32) -> i32 {
// FORCE-CREATE-CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// FORCE-CREATE-CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : i32
// FORCE-CREATE-CHECK:         %[[PRE_NEXT:.*]] = arith.addi %[[C0]], %[[C5]] : i32
// FORCE-CREATE-CHECK:         %[[PRE_COND:.*]] = arith.cmpi slt, %[[PRE_NEXT]], %[[BOUND]] : i32
// FORCE-CREATE-CHECK:         %[[IF:.*]] = scf.if %[[PRE_COND]] -> (i32) {
// FORCE-CREATE-CHECK:           %[[WHILE:.*]] = scf.while (%[[ARG1:.*]] = %[[PRE_NEXT]]) : (i32) -> i32 {
// FORCE-CREATE-CHECK:             %[[NEXT:.*]] = arith.addi %[[ARG1]], %[[C5]] : i32
// FORCE-CREATE-CHECK:             %[[COND:.*]] = arith.cmpi slt, %[[NEXT]], %[[BOUND]] : i32
// FORCE-CREATE-CHECK:             scf.condition(%[[COND]]) %[[NEXT]] : i32
// FORCE-CREATE-CHECK:           } do {
// FORCE-CREATE-CHECK:           ^bb0(%[[ARG2:.*]]: i32):
// FORCE-CREATE-CHECK:             scf.yield %[[ARG2]] : i32
// FORCE-CREATE-CHECK:           }
// FORCE-CREATE-CHECK:           scf.yield %[[WHILE]] : i32
// FORCE-CREATE-CHECK:         } else {
// FORCE-CREATE-CHECK:           scf.yield %[[PRE_NEXT]] : i32
// FORCE-CREATE-CHECK:         }
// FORCE-CREATE-CHECK:         return %[[IF]] : i32

// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-scf-uplift-while-to-for))' -split-input-file -allow-unregistered-dialect | FileCheck %s

func.func @uplift_while(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.test1"(%arg3) : (index) -> ()
    %added = arith.addi %arg3, %arg2 : index
    "test.test2"(%added) : (index) -> ()
    scf.yield %added : index
  }
  return %0 : index
}

// CHECK-LABEL: func @uplift_while
//  CHECK-SAME:     (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> index
//       CHECK:     %[[C1:.*]] = arith.constant 1 : index
//       CHECK:     scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] {
//       CHECK:     "test.test1"(%[[I]]) : (index) -> ()
//       CHECK:     %[[INC:.*]] = arith.addi %[[I]], %[[STEP]] : index
//       CHECK:     "test.test2"(%[[INC]]) : (index) -> ()
//       CHECK:     %[[R1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//       CHECK:     %[[R2:.*]] = arith.subi %[[END]], %[[BEGIN]] : index
//       CHECK:     %[[R3:.*]] = arith.addi %[[R2]], %[[R1]] : index
//       CHECK:     %[[R4:.*]] = arith.divsi %[[R3]], %[[STEP]] : index
//       CHECK:     %[[R5:.*]] = arith.subi %[[R4]], %[[C1]] : index
//       CHECK:     %[[R6:.*]] = arith.muli %[[R5]], %[[STEP]] : index
//       CHECK:     %[[R7:.*]] = arith.addi %[[BEGIN]], %[[R6]] : index
//       CHECK:     return %[[R7]] : index

// -----

func.func @uplift_while(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi sgt, %arg1, %arg3 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.test1"(%arg3) : (index) -> ()
    %added = arith.addi %arg3, %arg2 : index
    "test.test2"(%added) : (index) -> ()
    scf.yield %added : index
  }
  return %0 : index
}

// CHECK-LABEL: func @uplift_while
//  CHECK-SAME:     (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> index
//       CHECK:     %[[C1:.*]] = arith.constant 1 : index
//       CHECK:     scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] {
//       CHECK:     "test.test1"(%[[I]]) : (index) -> ()
//       CHECK:     %[[INC:.*]] = arith.addi %[[I]], %[[STEP]] : index
//       CHECK:     "test.test2"(%[[INC]]) : (index) -> ()
//       CHECK:     %[[R1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//       CHECK:     %[[R2:.*]] = arith.subi %[[END]], %[[BEGIN]] : index
//       CHECK:     %[[R3:.*]] = arith.addi %[[R2]], %[[R1]] : index
//       CHECK:     %[[R4:.*]] = arith.divsi %[[R3]], %[[STEP]] : index
//       CHECK:     %[[R5:.*]] = arith.subi %[[R4]], %[[C1]] : index
//       CHECK:     %[[R6:.*]] = arith.muli %[[R5]], %[[STEP]] : index
//       CHECK:     %[[R7:.*]] = arith.addi %[[BEGIN]], %[[R6]] : index
//       CHECK:     return %[[R7]] : index

// -----

func.func @uplift_while(%arg0: index, %arg1: index, %arg2: index) -> index {
  %0 = scf.while (%arg3 = %arg0) : (index) -> (index) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg3 : index
  } do {
  ^bb0(%arg3: index):
    "test.test1"(%arg3) : (index) -> ()
    %added = arith.addi %arg2, %arg3 : index
    "test.test2"(%added) : (index) -> ()
    scf.yield %added : index
  }
  return %0 : index
}

// CHECK-LABEL: func @uplift_while
//  CHECK-SAME:     (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> index
//       CHECK:     %[[C1:.*]] = arith.constant 1 : index
//       CHECK:     scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] {
//       CHECK:     "test.test1"(%[[I]]) : (index) -> ()
//       CHECK:     %[[INC:.*]] = arith.addi %[[STEP]], %[[I]] : index
//       CHECK:     "test.test2"(%[[INC]]) : (index) -> ()
//       CHECK:     %[[R1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//       CHECK:     %[[R2:.*]] = arith.subi %[[END]], %[[BEGIN]] : index
//       CHECK:     %[[R3:.*]] = arith.addi %[[R2]], %[[R1]] : index
//       CHECK:     %[[R4:.*]] = arith.divsi %[[R3]], %[[STEP]] : index
//       CHECK:     %[[R5:.*]] = arith.subi %[[R4]], %[[C1]] : index
//       CHECK:     %[[R6:.*]] = arith.muli %[[R5]], %[[STEP]] : index
//       CHECK:     %[[R7:.*]] = arith.addi %[[BEGIN]], %[[R6]] : index
//       CHECK:     return %[[R7]] : index


// -----

func.func @uplift_while(%arg0: index, %arg1: index, %arg2: index) -> (i32, f32) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2.0 : f32
  %0:3 = scf.while (%arg4 = %c1, %arg3 = %arg0, %arg5 = %c2) : (i32, index, f32) -> (i32, index, f32) {
    %1 = arith.cmpi slt, %arg3, %arg1 : index
    scf.condition(%1) %arg4, %arg3, %arg5 : i32, index, f32
  } do {
  ^bb0(%arg4: i32, %arg3: index, %arg5: f32):
    %1 = "test.test1"(%arg4) : (i32) -> i32
    %added = arith.addi %arg3, %arg2 : index
    %2 = "test.test2"(%arg5) : (f32) -> f32
    scf.yield %1, %added, %2 : i32, index, f32
  }
  return %0#0, %0#2 : i32, f32
}

// CHECK-LABEL: func @uplift_while
//  CHECK-SAME:     (%[[BEGIN:.*]]: index, %[[END:.*]]: index, %[[STEP:.*]]: index) -> (i32, f32)
//   CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:     %[[C2:.*]] = arith.constant 2.000000e+00 : f32
//       CHECK:     %[[RES:.*]]:2 = scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]]
//  CHECK-SAME:     iter_args(%[[ARG1:.*]] = %[[C1]], %[[ARG2:.*]] = %[[C2]]) -> (i32, f32) {
//       CHECK:     %[[T1:.*]] = "test.test1"(%[[ARG1]]) : (i32) -> i32
//       CHECK:     %[[T2:.*]] = "test.test2"(%[[ARG2]]) : (f32) -> f32
//       CHECK:     scf.yield %[[T1]], %[[T2]] : i32, f32
//       CHECK:     return %[[RES]]#0, %[[RES]]#1 : i32, f32

// -----

func.func @uplift_while(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
  %0 = scf.while (%arg3 = %arg0) : (i64) -> (i64) {
    %1 = arith.cmpi slt, %arg3, %arg1 : i64
    scf.condition(%1) %arg3 : i64
  } do {
  ^bb0(%arg3: i64):
    "test.test1"(%arg3) : (i64) -> ()
    %added = arith.addi %arg3, %arg2 : i64
    "test.test2"(%added) : (i64) -> ()
    scf.yield %added : i64
  }
  return %0 : i64
}

// CHECK-LABEL: func @uplift_while
//  CHECK-SAME:     (%[[BEGIN:.*]]: i64, %[[END:.*]]: i64, %[[STEP:.*]]: i64) -> i64
//       CHECK:     %[[C1:.*]] = arith.constant 1 : i64
//       CHECK:     scf.for %[[I:.*]] = %[[BEGIN]] to %[[END]] step %[[STEP]] : i64 {
//       CHECK:     "test.test1"(%[[I]]) : (i64) -> ()
//       CHECK:     %[[INC:.*]] = arith.addi %[[I]], %[[STEP]] : i64
//       CHECK:     "test.test2"(%[[INC]]) : (i64) -> ()
//       CHECK:     %[[R1:.*]] = arith.subi %[[STEP]], %[[C1]] : i64
//       CHECK:     %[[R2:.*]] = arith.subi %[[END]], %[[BEGIN]] : i64
//       CHECK:     %[[R3:.*]] = arith.addi %[[R2]], %[[R1]] : i64
//       CHECK:     %[[R4:.*]] = arith.divsi %[[R3]], %[[STEP]] : i64
//       CHECK:     %[[R5:.*]] = arith.subi %[[R4]], %[[C1]] : i64
//       CHECK:     %[[R6:.*]] = arith.muli %[[R5]], %[[STEP]] : i64
//       CHECK:     %[[R7:.*]] = arith.addi %[[BEGIN]], %[[R6]] : i64
//       CHECK:     return %[[R7]] : i64

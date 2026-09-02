// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-scf-prepare-uplift-while-to-for))' -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @test()
//       CHECK:  scf.while
//   CHECK-NOT:  "test.test1"
//       CHECK:  scf.condition(%{{.*}})
//       CHECK:  } do {
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  "test.test2"() : () -> ()
//       CHECK:  scf.yield
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  return
func.func @test() {
  scf.while () : () -> () {
    %1 = "test.cond"() : () -> i1
    "test.test1"() : () -> ()
    scf.condition(%1)
  } do {
  ^bb0():
    "test.test2"() : () -> ()
    scf.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @test()
//       CHECK:  scf.while
//   CHECK-NOT:  "test.test1"
//       CHECK:  scf.condition(%{{.*}})
//       CHECK:  } do {
//       CHECK:  %[[R1:.*]]:2 = "test.test1"() : () -> (i32, i64)
//       CHECK:  "test.test2"(%[[R1]]#1, %[[R1]]#0) : (i64, i32) -> ()
//       CHECK:  scf.yield
//       CHECK:  %[[R2:.*]]:2 = "test.test1"() : () -> (i32, i64)
//       CHECK:  return %[[R2]]#1, %[[R2]]#0 : i64, i32
func.func @test() -> (i64, i32) {
  %0:2 = scf.while () : () -> (i64, i32) {
    %1 = "test.cond"() : () -> i1
    %2:2 = "test.test1"() : () -> (i32, i64)
    scf.condition(%1) %2#1, %2#0 : i64, i32
  } do {
  ^bb0(%arg1: i64, %arg2: i32):
    "test.test2"(%arg1, %arg2) : (i64, i32) -> ()
    scf.yield
  }
  return %0#0, %0#1 : i64, i32
}

// -----

// CHECK-LABEL: func.func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: index)
//       CHECK:  %[[RES:.*]] = scf.while (%[[ARG1:.*]] = %[[ARG0]]) : (index) -> index {
//   CHECK-NOT:  arith.addi
//       CHECK:  scf.condition(%{{.*}}) %[[ARG1]] : index
//       CHECK:  } do {
//       CHECK:  ^bb0(%[[ARG2:.*]]: index):
//       CHECK:  %[[A1:.*]] = arith.addi %[[ARG0]], %[[ARG2]] : index
//       CHECK:  scf.yield %[[A1]]
//       CHECK:  %[[A2:.*]] = arith.addi %[[ARG0]], %[[RES]] : index
//       CHECK:  return %[[A2]]
func.func @test(%arg0: index) -> index {
  %res = scf.while (%arg1 = %arg0) : (index) -> (index) {
    %0 = arith.addi %arg0, %arg1 : index
    %1 = "test.cond"() : () -> i1
    scf.condition(%1) %0 : index
  } do {
  ^bb0(%arg2: index):
    scf.yield %arg2 : index
  }
  return %res : index
}

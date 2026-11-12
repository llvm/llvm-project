// RUN: mlir-opt %s -test-scf-while-op-builder | FileCheck %s

// CHECK-LABEL: @testMatchingTypes
func.func @testMatchingTypes(%arg0 : i32) {
  %0 = scf.while (%arg1 = %arg0) : (i32) -> (i32) {
    %c10 = arith.constant 10 : i32
    %1 = arith.cmpi slt, %arg1, %c10 : i32
    scf.condition(%1) %arg1 : i32
  } do {
  ^bb0(%arg1: i32):
    scf.yield %arg1 : i32
  }
  // Expect the same loop twice (the dummy added by the test pass and the
  // original one).
  // CHECK: %[[V0:.*]] = scf.while (%[[arg1:.*]] = %[[arg0:.*]]) : (i32) -> i32 {
  // CHECK: %[[V1:.*]] = scf.while (%[[arg2:.*]] = %[[arg0]]) : (i32) -> i32 {
  return
}

// CHECK-LABEL: @testNonMatchingTypes
func.func @testNonMatchingTypes(%arg0 : i32) {
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %0:2 = scf.while (%arg1 = %arg0) : (i32) -> (i32, i32) {
    %1 = arith.cmpi slt, %arg1, %c10 : i32
    scf.condition(%1) %arg1, %c1 : i32, i32
  } do {
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    scf.yield %1 : i32
  }
  // Expect the same loop twice (the dummy added by the test pass and the
  // original one).
  // CHECK: %[[V0:.*]] = scf.while (%[[arg1:.*]] = %[[arg0:.*]]) : (i32) -> (i32, i32) {
  // CHECK: %[[V1:.*]] = scf.while (%[[arg2:.*]] = %[[arg0]]) : (i32) -> (i32, i32) {
  return
}

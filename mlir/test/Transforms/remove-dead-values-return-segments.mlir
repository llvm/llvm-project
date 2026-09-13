// RUN: mlir-opt %s --split-input-file --remove-dead-values="canonicalize=0" | FileCheck %s
// RUN: mlir-opt %s --split-input-file --remove-dead-values="canonicalize=1" | FileCheck %s

// Remove unused function results and update the return operand segments.
// Check with and without canonicalization to test the cleanup pass directly.

// CHECK-LABEL: func.func private @remove_last_segment() -> i32 {
// CHECK-DAG: %[[V0:[^ ]+]] = arith.constant 11 : i32
// CHECK: "test.segmented_return"(%[[V0]]) <{operandSegmentSizes = array<i32: 1, 0>}> : (i32) -> ()
// CHECK-LABEL: func.func @caller()
// CHECK: call @remove_last_segment() : () -> i32
func.func private @remove_last_segment() -> (i32, i32) {
  %v0 = arith.constant 11 : i32
  %v1 = arith.constant 22 : i32
  "test.segmented_return"(%v0, %v1)
    <{operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
}
func.func @caller() -> i32 {
  %r:2 = func.call @remove_last_segment() : () -> (i32, i32)
  return %r#0 : i32
}

// -----

// CHECK-LABEL: func.func private @remove_first_segment() -> i32 {
// CHECK-DAG: %[[V1:[^ ]+]] = arith.constant 22 : i32
// CHECK: "test.segmented_return"(%[[V1]]) <{operandSegmentSizes = array<i32: 0, 1>}> : (i32) -> ()
// CHECK-LABEL: func.func @caller()
// CHECK: call @remove_first_segment() : () -> i32
func.func private @remove_first_segment() -> (i32, i32) {
  %v0 = arith.constant 11 : i32
  %v1 = arith.constant 22 : i32
  "test.segmented_return"(%v0, %v1)
    <{operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
}
func.func @caller() -> i32 {
  %r:2 = func.call @remove_first_segment() : () -> (i32, i32)
  return %r#1 : i32
}

// -----

// CHECK-LABEL: func.func private @shrink_both_segments() -> (i32, i32) {
// CHECK-DAG: %[[V0:[^ ]+]] = arith.constant 11 : i32
// CHECK-DAG: %[[V3:[^ ]+]] = arith.constant 44 : i32
// CHECK: "test.segmented_return"(%[[V0]], %[[V3]]) <{operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
// CHECK-LABEL: func.func @caller()
// CHECK: call @shrink_both_segments() : () -> (i32, i32)
func.func private @shrink_both_segments() -> (i32, i32, i32, i32) {
  %v0 = arith.constant 11 : i32
  %v1 = arith.constant 22 : i32
  %v2 = arith.constant 33 : i32
  %v3 = arith.constant 44 : i32
  "test.segmented_return"(%v0, %v1, %v2, %v3)
    <{operandSegmentSizes = array<i32: 2, 2>}> : (i32, i32, i32, i32) -> ()
}
func.func @caller() -> (i32, i32) {
  %r:4 = func.call @shrink_both_segments() : () -> (i32, i32, i32, i32)
  return %r#0, %r#3 : i32, i32
}

// -----

// CHECK-LABEL: func.func private @remove_all_results() {
// CHECK: "test.segmented_return"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> ()
// CHECK-LABEL: func.func @caller()
// CHECK: call @remove_all_results() : () -> ()
func.func private @remove_all_results() -> (i32, i32) {
  %v0 = arith.constant 11 : i32
  %v1 = arith.constant 22 : i32
  "test.segmented_return"(%v0, %v1)
    <{operandSegmentSizes = array<i32: 1, 1>}> : (i32, i32) -> ()
}
func.func @caller() {
  %r:2 = func.call @remove_all_results() : () -> (i32, i32)
  return
}

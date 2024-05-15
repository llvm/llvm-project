// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s

// CHECK-LABEL: @extsiOnI1
//       CHECK: %[[TRUE:.*]] = arith.constant true
//       CHECK: %[[CST:.*]] = arith.constant -1 : i16
//       CHECK: return %[[TRUE]], %[[CST]]
func.func @extsiOnI1() -> (i1, i16) {
  %true = arith.constant -1 : i1
  %0 = arith.extsi %true : i1 to i16
  return %true, %0 : i1, i16
}

// CHECK-LABEL: @extuiOn1I1
//       CHECK: %[[TRUE:.*]] = arith.constant true
//       CHECK: %[[CST:.*]] = arith.constant 1 : i64
//       CHECK: return %[[TRUE]], %[[CST]]
func.func @extuiOn1I1() -> (i1, i64) {
  %true = arith.constant true
  %0 = arith.extui %true : i1 to i64
  return %true, %0 : i1, i64
}

// CHECK-LABEL: @trunciI16ToI8
//       CHECK: %[[CST:.*]] = arith.constant 20194 : i16
//       CHECK: %[[CST2:.*]] = arith.constant -30 : i8
//       CHECK: return %[[CST]], %[[CST2]]
func.func @trunciI16ToI8() -> (i16, i8) {
  %c20194_i16 = arith.constant 20194 : i16
  %0 = arith.trunci %c20194_i16 : i16 to i8
  return %c20194_i16, %0 : i16, i8
}
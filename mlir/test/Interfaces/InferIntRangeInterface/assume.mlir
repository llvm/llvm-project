// RUN: mlir-opt %s -int-range-optimizations -split-input-file | FileCheck %s

// The fact applies to the result, but not to the original predicated value.
// CHECK-LABEL: func.func @result_scope(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i32) -> (i1, i1) {
//   CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[TRUE:.*]] = arith.constant true
//       CHECK:   %[[ORIGINAL:.*]] = arith.cmpi sgt, %[[VALUE]], %[[ZERO]] : i32
//  CHECK-NEXT:   return %[[TRUE]], %[[ORIGINAL]] : i1, i1
func.func @result_scope(%value: i32) -> (i1, i1) {
  %zero = arith.constant 0 : i32
  %positive = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  %positive_cmp = arith.cmpi sgt, %positive, %zero : i32
  %original_cmp = arith.cmpi sgt, %value, %zero : i32
  return %positive_cmp, %original_cmp : i1, i1
}

// -----

// CHECK-LABEL: func.func @intersect_assumptions(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i8) -> i8 {
//  CHECK-NEXT:   %[[LOWER:.*]] = ub.assume %[[VALUE]], #arith.assume_cmpi<sge, 4 : i8>() : i8
//  CHECK-NEXT:   %[[BOUNDED:.*]] = ub.assume %[[LOWER]], #arith.assume_cmpi<slt, 10 : i8>() : i8
//  CHECK-NEXT:   %[[RESULT:.*]] = test.reflect_bounds
//  CHECK-SAME:       <umin = 4 : ui8, umax = 9 : ui8,
//  CHECK-SAME:        smin = 4 : si8, smax = 9 : si8> %[[BOUNDED]] : i8
//  CHECK-NEXT:   return %[[RESULT]] : i8
func.func @intersect_assumptions(%value: i8) -> i8 {
  %lower = ub.assume %value, #arith.assume_cmpi<sge, 4 : i8>() : i8
  %bounded = ub.assume %lower, #arith.assume_cmpi<slt, 10 : i8>() : i8
  %result = test.reflect_bounds %bounded : i8
  return %result : i8
}

// -----

// CHECK-LABEL: func.func @exclude_zero(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i8) -> i8 {
//  CHECK-NEXT:   %[[BOUNDED:.*]] = ub.assume %[[VALUE]], #arith.assume_cmpi<ne, 0 : i8>() : i8
//  CHECK-NEXT:   %[[RESULT:.*]] = test.reflect_bounds
//  CHECK-SAME:       <umin = 1 : ui8, umax = 255 : ui8,
//  CHECK-SAME:        smin = -128 : si8, smax = 127 : si8> %[[BOUNDED]] : i8
//  CHECK-NEXT:   return %[[RESULT]] : i8
func.func @exclude_zero(%value: i8) -> i8 {
  %bounded = ub.assume %value, #arith.assume_cmpi<ne, 0 : i8>() : i8
  %result = test.reflect_bounds %bounded : i8
  return %result : i8
}

// -----

// A UB result must be materialized as an arith constant if its range is a
// singleton. The predicated value must remain unconstrained.
// CHECK-LABEL: func.func @singleton(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i128) -> (i128, i128) {
//  CHECK-NEXT:   %[[C42:.*]] = arith.constant 42 : i128
//  CHECK-NEXT:   return %[[C42]], %[[VALUE]] : i128, i128
func.func @singleton(%value: i128) -> (i128, i128) {
  %result = ub.assume %value, #arith.assume_cmpi<eq, 42 : i128>() : i128
  return %result, %value : i128, i128
}

// -----

// Contradictions must not publish inverted bounds or imply unreachable code.
// CHECK-LABEL: func.func @contradiction(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i8) -> i8 {
//  CHECK-NEXT:   %[[POSITIVE:.*]] = ub.assume %[[VALUE]], #arith.assume_cmpi<sgt, 0 : i8>() : i8
//  CHECK-NEXT:   %[[NEGATIVE:.*]] = ub.assume %[[POSITIVE]], #arith.assume_cmpi<slt, 0 : i8>() : i8
//  CHECK-NEXT:   %[[RESULT:.*]] = test.reflect_bounds
//  CHECK-SAME:       <umin = 0 : ui8, umax = 255 : ui8,
//  CHECK-SAME:        smin = -128 : si8, smax = 127 : si8> %[[NEGATIVE]] : i8
//  CHECK-NEXT:   return %[[RESULT]] : i8
func.func @contradiction(%value: i8) -> i8 {
  %positive = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i8>() : i8
  %negative = ub.assume %positive, #arith.assume_cmpi<slt, 0 : i8>() : i8
  %result = test.reflect_bounds %negative : i8
  return %result : i8
}

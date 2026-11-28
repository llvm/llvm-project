// RUN: mlir-opt %s -canonicalize="test-convergence" --split-input-file | FileCheck %s


// CHECK-LABEL: func @merge_poison()
//       CHECK:   %[[RES:.*]] = ub.poison : i32
//       CHECK:   return %[[RES]], %[[RES]]
func.func @merge_poison() -> (i32, i32) {
  %0 = ub.poison : i32
  %1 = ub.poison : i32
  return %0, %1 : i32, i32
}

// -----

// CHECK-LABEL: func.func @assume_true() -> i32 {
//  CHECK-NEXT:   %[[VALUE:.*]] = arith.constant 7 : i32
//  CHECK-NEXT:   return %[[VALUE]] : i32
func.func @assume_true() -> i32 {
  %value = arith.constant 7 : i32
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  return %assumed : i32
}

// -----

// CHECK-LABEL: func.func @assume_false() -> i32 {
//  CHECK-NEXT:   %[[POISON:.*]] = ub.poison : i32
//  CHECK-NEXT:   return %[[POISON]] : i32
func.func @assume_false() -> i32 {
  %value = arith.constant 0 : i32
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  return %assumed : i32
}

// -----

// CHECK-LABEL: func.func @assume_poison() -> i32 {
//  CHECK-NEXT:   %[[POISON:.*]] = ub.poison : i32
//  CHECK-NEXT:   return %[[POISON]] : i32
func.func @assume_poison() -> i32 {
  %value = ub.poison : i32
  %assumed = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  return %assumed : i32
}

// -----

// CHECK-LABEL: func.func @assume_repeated(
//  CHECK-SAME: %[[VALUE:[^:]+]]: i32) -> i32 {
//  CHECK-NEXT:   %[[ASSUMED:.*]] = ub.assume %[[VALUE]], #arith.assume_cmpi<sgt, 0 : i32>() : i32
//  CHECK-NEXT:   return %[[ASSUMED]] : i32
func.func @assume_repeated(%value: i32) -> i32 {
  %first = ub.assume %value, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  %second = ub.assume %first, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  return %second : i32
}

// -----

// A failed assumption can be masked without causing immediate UB.
// CHECK-LABEL: func.func @masked_false_assumption() -> i32 {
//  CHECK-NEXT:   %[[VALUE:.*]] = arith.constant 7 : i32
//  CHECK-NEXT:   return %[[VALUE]] : i32
func.func @masked_false_assumption() -> i32 {
  %false = arith.constant false
  %zero = arith.constant 0 : i32
  %value = arith.constant 7 : i32
  %assumed = ub.assume %zero, #arith.assume_cmpi<sgt, 0 : i32>() : i32
  %selected = arith.select %false, %assumed, %value : i32
  return %selected : i32
}

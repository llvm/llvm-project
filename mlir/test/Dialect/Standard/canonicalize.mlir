// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @select_same_val
//       CHECK:   return %arg1
func @select_same_val(%arg0: i1, %arg1: i64) -> i64 {
  %0 = arith.select %arg0, %arg1, %arg1 : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @select_cmp_ne_select
//       CHECK:   return %arg0
func @select_cmp_ne_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  %1 = arith.select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @select_extui
//       CHECK:   %[[res:.+]] = arith.extui %arg0 : i1 to i64
//       CHECK:   return %[[res]]
func @select_extui(%arg0: i1) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %res = arith.select %arg0, %c1_i64, %c0_i64 : i64
  return %res : i64
}

// CHECK-LABEL: @select_extui2
// CHECK-DAG:  %true = arith.constant true
// CHECK-DAG:  %[[xor:.+]] = arith.xori %arg0, %true : i1
// CHECK-DAG:  %[[res:.+]] = arith.extui %[[xor]] : i1 to i64
//       CHECK:   return %[[res]]
func @select_extui2(%arg0: i1) -> i64 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %res = arith.select %arg0, %c0_i64, %c1_i64 : i64
  return %res : i64
}

// -----

// CHECK-LABEL: @select_extui_i1
//  CHECK-NEXT:   return %arg0
func @select_extui_i1(%arg0: i1) -> i1 {
  %c0_i1 = arith.constant false
  %c1_i1 = arith.constant true
  %res = arith.select %arg0, %c1_i1, %c0_i1 : i1
  return %res : i1
}

// -----

// CHECK-LABEL: @selToNot
//       CHECK:       %[[trueval:.+]] = arith.constant true
//       CHECK:       %[[res:.+]] = arith.xori %arg0, %[[trueval]] : i1
//       CHECK:   return %[[res]]
func @selToNot(%arg0: i1) -> i1 {
  %true = arith.constant true
  %false = arith.constant false
  %res = arith.select %arg0, %false, %true : i1
  return %res : i1
}

// CHECK-LABEL: @selToArith
//       CHECK-NEXT:       %[[trueval:.+]] = arith.constant true
//       CHECK-NEXT:       %[[notcmp:.+]] = arith.xori %arg0, %[[trueval]] : i1
//       CHECK-NEXT:       %[[condtrue:.+]] = arith.andi %arg0, %arg1 : i1
//       CHECK-NEXT:       %[[condfalse:.+]] = arith.andi %[[notcmp]], %arg2 : i1
//       CHECK-NEXT:       %[[res:.+]] = arith.ori %[[condtrue]], %[[condfalse]] : i1
//       CHECK:   return %[[res]]
func @selToArith(%arg0: i1, %arg1 : i1, %arg2 : i1) -> i1 {
  %res = arith.select %arg0, %arg1, %arg2 : i1
  return %res : i1
}

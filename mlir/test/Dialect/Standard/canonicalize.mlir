// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// CHECK-LABEL: @select_same_val
//       CHECK:   return %arg1
func @select_same_val(%arg0: i1, %arg1: i64) -> i64 {
  %0 = select %arg0, %arg1, %arg1 : i64
  return %0 : i64
}

// -----

// CHECK-LABEL: @select_cmp_eq_select
//       CHECK:   return %arg1
func @select_cmp_eq_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi eq, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @select_cmp_ne_select
//       CHECK:   return %arg0
func @select_cmp_ne_select(%arg0: i64, %arg1: i64) -> i64 {
  %0 = arith.cmpi ne, %arg0, %arg1 : i64
  %1 = select %0, %arg0, %arg1 : i64
  return %1 : i64
}

// -----

// CHECK-LABEL: @branchCondProp
//       CHECK:       %[[trueval:.+]] = arith.constant true
//       CHECK:       %[[falseval:.+]] = arith.constant false
//       CHECK:       "test.consumer1"(%[[trueval]]) : (i1) -> ()
//       CHECK:       "test.consumer2"(%[[falseval]]) : (i1) -> ()
func @branchCondProp(%arg0: i1) {
  cond_br %arg0, ^trueB, ^falseB

^trueB:
  "test.consumer1"(%arg0) : (i1) -> ()
  br ^exit

^falseB:
  "test.consumer2"(%arg0) : (i1) -> ()
  br ^exit

^exit:
  return
}

// -----

// CHECK-LABEL: @selToNot
//       CHECK:       %[[trueval:.+]] = arith.constant true
//       CHECK:       %{{.+}} = arith.xori %arg0, %[[trueval]] : i1
func @selToNot(%arg0: i1) -> i1 {
  %true = arith.constant true
  %false = arith.constant false
  %res = select %arg0, %false, %true : i1
  return %res : i1
}

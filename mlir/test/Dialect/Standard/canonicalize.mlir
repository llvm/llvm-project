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

// CHECK-LABEL: test_maxsi
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MAX_INT_CST:.+]] = arith.constant 127
// CHECK: %[[X:.+]] = maxsi %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func @test_maxsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = maxsi %arg0, %arg0 : i8
  %1 = maxsi %arg0, %maxIntCst : i8
  %2 = maxsi %arg0, %minIntCst : i8
  %3 = maxsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_maxui
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MAX_INT_CST:.+]] = arith.constant -1
// CHECK: %[[X:.+]] = maxui %arg0, %[[C0]]
// CHECK: return %arg0, %[[MAX_INT_CST]], %arg0, %[[X]]
func @test_maxui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = maxui %arg0, %arg0 : i8
  %1 = maxui %arg0, %maxIntCst : i8
  %2 = maxui %arg0, %minIntCst : i8
  %3 = maxui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}


// CHECK-LABEL: test_minsi
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MIN_INT_CST:.+]] = arith.constant -128
// CHECK: %[[X:.+]] = minsi %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func @test_minsi(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 127 : i8
  %minIntCst = arith.constant -128 : i8
  %c0 = arith.constant 42 : i8
  %0 = minsi %arg0, %arg0 : i8
  %1 = minsi %arg0, %maxIntCst : i8
  %2 = minsi %arg0, %minIntCst : i8
  %3 = minsi %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// CHECK-LABEL: test_minui
// CHECK: %[[C0:.+]] = arith.constant 42
// CHECK: %[[MIN_INT_CST:.+]] = arith.constant 0
// CHECK: %[[X:.+]] = minui %arg0, %[[C0]]
// CHECK: return %arg0, %arg0, %[[MIN_INT_CST]], %[[X]]
func @test_minui(%arg0 : i8) -> (i8, i8, i8, i8) {
  %maxIntCst = arith.constant 255 : i8
  %minIntCst = arith.constant 0 : i8
  %c0 = arith.constant 42 : i8
  %0 = minui %arg0, %arg0 : i8
  %1 = minui %arg0, %maxIntCst : i8
  %2 = minui %arg0, %minIntCst : i8
  %3 = minui %arg0, %c0 : i8
  return %0, %1, %2, %3: i8, i8, i8, i8
}

// RUN: mlir-opt %s -test-greedy-patterns="max-iterations=1 top-down=true" \
// RUN:     --split-input-file | FileCheck %s

// Tests for https://github.com/llvm/llvm-project/issues/86765. Ensure
// that operands of a dead op are added to the worklist even if the same value
// appears multiple times as an operand.

// 2 uses of the same operand

// CHECK:       func.func @f(%arg0: i1) {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @f(%arg0: i1) {
  %0 = arith.constant 0 : i32
  %if = scf.if %arg0 -> (i32) {
    scf.yield %0 : i32
  } else {
    scf.yield %0 : i32
  }
  %dead_leaf = arith.addi %if, %if : i32
  return
}

// -----

// 3 uses of the same operand

// CHECK:       func.func @f() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @f() {
  %0 = arith.constant 0 : i1
  %if = scf.if %0 -> (i1) {
    scf.yield %0 : i1
  } else {
    scf.yield %0 : i1
  }
  %dead_leaf = arith.select %if, %if, %if : i1
  return
}

// -----

// 2 uses of the same operand, op has 3 operands

// CHECK:       func.func @f() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @f() {
  %0 = arith.constant 0 : i1
  %if = scf.if %0 -> (i1) {
    scf.yield %0 : i1
  } else {
    scf.yield %0 : i1
  }
  %dead_leaf = arith.select %0, %if, %if : i1
  return
}

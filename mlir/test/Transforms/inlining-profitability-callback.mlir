// RUN: mlir-opt %s -test-inline-profitability-callback | FileCheck %s

func.func @callee() -> i32 {
  %0 = arith.constant 0 : i32
  return %0 : i32
}

// CHECK-LABEL: func @inline_profitable
func.func @inline_profitable() -> i32 {
  // CHECK-NEXT: arith.constant
  // CHECK-NEXT: return
  %0 = call @callee() : () -> i32
  return %0 : i32
}

// CHECK-LABEL: func @no_inline_unprofitable
func.func @no_inline_unprofitable() -> i32 {
  // CHECK-NEXT: call @callee
  // CHECK-NEXT: return
  %0 = call @callee() {unprofitable} : () -> i32
  return %0 : i32
}

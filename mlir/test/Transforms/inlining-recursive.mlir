// RUN: mlir-opt %s -inline='default-pipeline=' | FileCheck %s
// RUN: mlir-opt %s --mlir-disable-threading -inline='default-pipeline=' | FileCheck %s

// CHECK-LABEL: func.func @foo0
func.func @foo0(%arg0 : i32) -> i32 {
  // CHECK: call @foo1
  // CHECK: }
  %0 = arith.constant 0 : i32
  %1 = arith.cmpi eq, %arg0, %0 : i32
  cf.cond_br %1, ^exit, ^tail
^exit:
  return %0 : i32
^tail:
  %3 = call @foo1(%arg0) : (i32) -> i32
  return %3 : i32
}

// CHECK-LABEL: func.func @foo1
func.func @foo1(%arg0 : i32) -> i32 {
  // CHECK:    call @foo0
  %0 = arith.constant 1 : i32
  %1 = arith.subi %arg0, %0 : i32
  %2 = call @foo0(%1) : (i32) -> i32
  return %2 : i32
}

// RUN: mlir-opt %s -inline='default-pipeline= inlining-threshold=100' | FileCheck %s

// Check that inlining does not happen when the threshold is exceeded.
func.func @callee1(%arg : i32) -> i32 {
  %v1 = arith.addi %arg, %arg : i32
  %v2 = arith.addi %v1, %arg : i32
  %v3 = arith.addi %v2, %arg : i32
  return %v3 : i32
}

// CHECK-LABEL: func @caller1
func.func @caller1(%arg0 : i32) -> i32 {
  // CHECK-NEXT: call @callee1
  // CHECK-NEXT: return

  %0 = call @callee1(%arg0) : (i32) -> i32
  return %0 : i32
}

// RUN: not mlir-opt %s -split-input-file --test-bytecode-roundtrip="test-kind=5" 2>&1 | FileCheck %s

// CHECK-NOT: failed to read bytecode
func.func @base_test(%arg0 : i32) -> f32 {
  %0 = "test.addi"(%arg0, %arg0) : (i32, i32) -> i32
  %1 = "test.cast"(%0) : (i32) -> f32
  return %1 : f32
}

// -----

// CHECK-LABEL: error: unknown attribute code: 99
// CHECK: failed to read bytecode
func.func @base_test(%arg0 : !test.i32) -> f32 {
  %0 = "test.addi"(%arg0, %arg0) : (!test.i32, !test.i32) -> !test.i32
  %1 = "test.cast"(%0) : (!test.i32) -> f32
  return %1 : f32
}

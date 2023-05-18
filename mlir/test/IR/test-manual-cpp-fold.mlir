// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

func.func @test() -> i32 {
  %c5 = "test.constant"() {value = 5 : i32} : () -> i32
  %res = "test.manual_cpp_op_with_fold"(%c5) : (i32) -> i32
  return %res : i32
}

// CHECK-LABEL: func.func @test
// CHECK-NEXT: %[[C:.*]] = "test.constant"() <{value = 5 : i32}>
// CHECK-NEXT: return %[[C]]

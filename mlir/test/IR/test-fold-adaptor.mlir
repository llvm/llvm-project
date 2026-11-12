// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

func.func @test() -> i32 {
  %c5 = "test.constant"() {value = 5 : i32} : () -> i32
  %c1 = "test.constant"() {value = 1 : i32} : () -> i32
  %c2 = "test.constant"() {value = 2 : i32} : () -> i32
  %c3 = "test.constant"() {value = 3 : i32} : () -> i32
  %res = test.fold_with_fold_adaptor %c5, [ %c1, %c2], { (%c3), (%c3) } {
    %c0 = "test.constant"() {value = 0 : i32} : () -> i32
  }
  return %res : i32
}

// CHECK-LABEL: func.func @test
// CHECK-NEXT: %[[C:.*]] = "test.constant"() <{value = 33 : i32}>
// CHECK-NEXT: return %[[C]]

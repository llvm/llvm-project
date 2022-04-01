// RUN: mlir-opt -test-patterns %s | FileCheck %s

func @foo() -> i32 {
  %c42 = arith.constant 42 : i32

  // The new operation should be present in the output and contain an attribute
  // with value "42" that results from folding.

  // CHECK: "test.op_in_place_fold"(%{{.*}}) {attr = 42 : i32}
  %0 = "test.op_in_place_fold_anchor"(%c42) : (i32) -> (i32)
  return %0 : i32
}

func @test_fold_before_previously_folded_op() -> (i32, i32) {
  // When folding two constants will be generated and uniqued. Check that the
  // uniqued constant properly dominates both uses.
  // CHECK: %[[CST:.+]] = arith.constant true
  // CHECK-NEXT: "test.cast"(%[[CST]]) : (i1) -> i32
  // CHECK-NEXT: "test.cast"(%[[CST]]) : (i1) -> i32

  %0 = "test.cast"() {test_fold_before_previously_folded_op} : () -> (i32)
  %1 = "test.cast"() {test_fold_before_previously_folded_op} : () -> (i32)
  return %0, %1 : i32, i32
}

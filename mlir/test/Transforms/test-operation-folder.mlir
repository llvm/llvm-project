// RUN: mlir-opt -test-patterns='top-down=false' %s | FileCheck %s
// RUN: mlir-opt -test-patterns='top-down=true' %s | FileCheck %s

func.func @foo() -> i32 {
  %c42 = arith.constant 42 : i32

  // The new operation should be present in the output and contain an attribute
  // with value "42" that results from folding.

  // CHECK: "test.op_in_place_fold"(%{{.*}}) {attr = 42 : i32}
  %0 = "test.op_in_place_fold_anchor"(%c42) : (i32) -> (i32)
  return %0 : i32
}

func.func @test_fold_before_previously_folded_op() -> (i32, i32) {
  // When folding two constants will be generated and uniqued. Check that the
  // uniqued constant properly dominates both uses.
  // CHECK: %[[CST:.+]] = arith.constant true
  // CHECK-NEXT: "test.cast"(%[[CST]]) : (i1) -> i32
  // CHECK-NEXT: "test.cast"(%[[CST]]) : (i1) -> i32

  %0 = "test.cast"() {test_fold_before_previously_folded_op} : () -> (i32)
  %1 = "test.cast"() {test_fold_before_previously_folded_op} : () -> (i32)
  return %0, %1 : i32, i32
}

func.func @test_dont_reorder_constants() -> (i32, i32, i32) {
  // Test that we don't reorder existing constants during folding if it isn't necessary.
  // CHECK: %[[CST:.+]] = arith.constant 1
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 2
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 3
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  return %0, %1, %2 : i32, i32, i32
}

// RUN: mlir-opt -test-greedy-patterns='top-down=false' %s | FileCheck %s
// RUN: mlir-opt -test-greedy-patterns='top-down=true' %s | FileCheck %s
// RUN: mlir-opt -test-greedy-patterns='cse-constants=false' %s | FileCheck %s --check-prefix=NOCSE
// RUN: mlir-opt -test-greedy-patterns='fold=false' %s | FileCheck %s --check-prefix=NOFOLD

func.func @foo() -> i32 {
  %c42 = arith.constant 42 : i32

  // The new operation should be present in the output and contain an attribute
  // with value "42" that results from folding.

  // CHECK: "test.op_in_place_fold"(%{{.*}}) <{attr = 42 : i32}
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
  // Test that we don't reorder existing constants during folding if it isn't
  // necessary.
  // CHECK: %[[CST:.+]] = arith.constant 1
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 2
  // CHECK-NEXT: %[[CST:.+]] = arith.constant 3
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  return %0, %1, %2 : i32, i32, i32
}

// CHECK-LABEL: test_fold_nofold_nocse
// NOCSE-LABEL: test_fold_nofold_nocse
// NOFOLD-LABEL: test_fold_nofold_nocse
func.func @test_fold_nofold_nocse() -> (i32, i32, i32, i32, i32, i32) {
  // Test either not folding or deduping constants.

  // Testing folding. There should be only 4 constants here.
  // CHECK-NOT: arith.constant
  // CHECK-DAG: %[[CST:.+]] = arith.constant 0
  // CHECK-DAG: %[[CST:.+]] = arith.constant 1
  // CHECK-DAG: %[[CST:.+]] = arith.constant 2
  // CHECK-DAG: %[[CST:.+]] = arith.constant 3
  // CHECK-NOT: arith.constant
  // CHECK-NEXT: return

  // Testing not-CSE'ing. In this case we have the 3 original constants and 3
  // produced by folding.
  // NOCSE-DAG: arith.constant 0 : i32
  // NOCSE-DAG: arith.constant 1 : i32
  // NOCSE-DAG: arith.constant 2 : i32
  // NOCSE-DAG: arith.constant 1 : i32
  // NOCSE-DAG: arith.constant 2 : i32
  // NOCSE-DAG: arith.constant 3 : i32
  // NOCSE-NEXT: return

  // Testing not folding. In this case we just have the original constants.
  // NOFOLD-DAG: %[[CST:.+]] = arith.constant 0
  // NOFOLD-DAG: %[[CST:.+]] = arith.constant 1
  // NOFOLD-DAG: %[[CST:.+]] = arith.constant 2
  // NOFOLD: arith.addi
  // NOFOLD: arith.addi
  // NOFOLD: arith.addi

  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %0 = arith.addi %c0, %c1 : i32
  %1 = arith.addi %0, %c1 : i32
  %2 = arith.addi %c2, %c1 : i32
  return %0, %1, %2, %c0, %c1, %c2 : i32, i32, i32, i32, i32, i32
}


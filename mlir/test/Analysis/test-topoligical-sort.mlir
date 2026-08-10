// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-print-topological-sort))" --split-input-file | FileCheck %s

// CHECK-LABEL: single_element
func.func @single_element() {
  // CHECK: test_sort_index = 0
  return {test_to_sort}
}

// -----

// CHECK-LABEL: @simple_region
func.func @simple_region(%cond: i1) {
  // CHECK: test_sort_index = 0
  %0 = arith.constant {test_to_sort} 42 : i32
  scf.if %cond {
    %1 = arith.addi %0, %0 : i32
    // CHECK: test_sort_index = 2
    %2 = arith.subi %0, %1 {test_to_sort} : i32
  // CHECK: test_sort_index = 1
  } {test_to_sort}
  return
}

// -----

// CHECK-LABEL: @multi_region
func.func @multi_region(%cond: i1) {
  scf.if %cond {
    // CHECK: test_sort_index = 0
    %0 = arith.constant {test_to_sort} 42 : i32
  }

  scf.if %cond {
    // CHECK: test_sort_index = 1
    %0 = arith.constant {test_to_sort} 24 : i32
  }
  return
}

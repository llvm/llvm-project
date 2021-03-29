// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @remove_op_with_inner_ops_pattern
func @remove_op_with_inner_ops_pattern() {
  // CHECK-NEXT: return
  "test.op_with_region_pattern"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold_no_side_effect
func @remove_op_with_inner_ops_fold_no_side_effect() {
  // CHECK-NEXT: return
  "test.op_with_region_fold_no_side_effect"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func @remove_op_with_inner_ops_fold(%arg0 : i32) -> (i32) {
  // CHECK-NEXT: return %[[ARG_0]]
  %0 = "test.op_with_region_fold"(%arg0) ({
    "test.op_with_region_terminator"() : () -> ()
  }) : (i32) -> (i32)
  return %0 : i32
}

// CHECK-LABEL: func @remove_op_with_variadic_results_and_folder
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func @remove_op_with_variadic_results_and_folder(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK-NEXT: return %[[ARG_0]], %[[ARG_1]]
  %0, %1 = "test.op_with_variadic_results_and_folder"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// CHECK-LABEL: func @test_commutative_multi
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func @test_commutative_multi(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK: %[[C42:.*]] = constant 42 : i32
  %c42_i32 = constant 42 : i32
  // CHECK: %[[C43:.*]] = constant 43 : i32
  %c43_i32 = constant 43 : i32
  // CHECK-NEXT: %[[O0:.*]] = "test.op_commutative"(%[[ARG_0]], %[[ARG_1]], %[[C42]], %[[C43]]) : (i32, i32, i32, i32) -> i32
  %y = "test.op_commutative"(%c42_i32, %arg0, %arg1, %c43_i32) : (i32, i32, i32, i32) -> i32

  // CHECK-NEXT: %[[O1:.*]] = "test.op_commutative"(%[[ARG_0]], %[[ARG_1]], %[[C42]], %[[C43]]) : (i32, i32, i32, i32) -> i32
  %z = "test.op_commutative"(%arg0, %c42_i32, %c43_i32, %arg1): (i32, i32, i32, i32) -> i32
  // CHECK-NEXT: return %[[O0]], %[[O1]]
  return %y, %z: i32, i32
}


// CHECK-LABEL: func @test_commutative_multi_cst
func @test_commutative_multi_cst(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %c42_i32 = constant 42 : i32
  %c42_i32 = constant 42 : i32
  %c42_i32_2 = constant 42 : i32
  // CHECK-NEXT: %[[O0:.*]] = "test.op_commutative"(%arg0, %arg1, %c42_i32, %c42_i32) : (i32, i32, i32, i32) -> i32
  %y = "test.op_commutative"(%c42_i32, %arg0, %arg1, %c42_i32_2) : (i32, i32, i32, i32) -> i32

  %c42_i32_3 = constant 42 : i32

  // CHECK-NEXT: %[[O1:.*]] = "test.op_commutative"(%arg0, %arg1, %c42_i32, %c42_i32) : (i32, i32, i32, i32) -> i32
  %z = "test.op_commutative"(%arg0, %c42_i32_3, %c42_i32_2, %arg1): (i32, i32, i32, i32) -> i32
  // CHECK-NEXT: return %[[O0]], %[[O1]]
  return %y, %z: i32, i32
}

// CHECK-LABEL: func @typemismatch

func @typemismatch() -> i32 {
  %c42 = constant 42.0 : f32

  // The "passthrough_fold" folder will naively return its operand, but we don't
  // want to fold here because of the type mismatch.

  // CHECK: "test.passthrough_fold"
  %0 = "test.passthrough_fold"(%c42) : (f32) -> (i32)
  return %0 : i32
}

// CHECK-LABEL: func @result_shape_per_dim
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: tensor<2x3x?xf32>, %[[ARG_1:[a-z0-9]*]]: tensor<?x5xf32>)
func @result_shape_per_dim(%arg0 : tensor<2x3x?xf32>, %arg1 : tensor<?x5xf32>)
    -> (index, index, index, index, index) {
  // CHECK-DAG: %[[C0:.+]] = constant 0 : index
  // CHECK-DAG: %[[C2:.+]] = constant 2 : index
  // CHECK-DAG: %[[C3:.+]] = constant 3 : index
  // CHECK-DAG: %[[C5:.+]] = constant 5 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0:2 = "test.op_with_result_shape_per_dim_interface"(%arg0, %arg1)
      : (tensor<2x3x?xf32>, tensor<?x5xf32>) -> (tensor<?x5xf32>, tensor<2x3x?xf32>)
  %1 = memref.dim %0#0, %c0 : tensor<?x5xf32>
  %2 = memref.dim %0#0, %c1 : tensor<?x5xf32>
  %3 = memref.dim %0#1, %c0 : tensor<2x3x?xf32>
  %4 = memref.dim %0#1, %c1 : tensor<2x3x?xf32>
  %5 = memref.dim %0#1, %c2 : tensor<2x3x?xf32>
  // CHECK-DAG: %[[D0:.+]] = memref.dim %[[ARG_1]], %[[C0]]
  // CHECK-DAG: %[[D1:.+]] = memref.dim %[[ARG_0]], %[[C2]]
  // CHECK: return %[[D0]], %[[C5]], %[[C2]], %[[C3]], %[[D1]]
  return %1, %2, %3, %4, %5 : index, index, index, index, index
}
// RUN: mlir-opt -test-grid-simplifications %s | FileCheck %s

shard.grid @grid0(shape = 4x2)
shard.grid @grid1(shape = 4)

// Checks that `all_reduce(x) + all_reduce(y)` gets transformed to
// `all_reduce(x + y)`.
// CHECK-LABEL: func.func @all_reduce_arith_addf_endomorphism
func.func @all_reduce_arith_addf_endomorphism(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ADD_RES]]
  // CHECK: return %[[ALL_REDUCE_RES]]
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_addf_endomorphism_multiple_uses_of_result
func.func @all_reduce_arith_addf_endomorphism_multiple_uses_of_result(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ADD_RES]]
  // CHECK: return %[[ALL_REDUCE_RES]], %[[ALL_REDUCE_RES]]
  return %2, %2 : tensor<5xf32>, tensor<5xf32>
}

// Do not simplify if there is another use of one of the all-reduces.
// CHECK-LABEL: func.func @all_reduce_arith_addf_endomorphism_multiple_uses_of_all_reduce_result
func.func @all_reduce_arith_addf_endomorphism_multiple_uses_of_all_reduce_result(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
  // CHECK: %[[ALL_REDUCE_0_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG0]]
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ALL_REDUCE_1_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG1]]
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ALL_REDUCE_0_RES]], %[[ALL_REDUCE_1_RES]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: return %[[ALL_REDUCE_0_RES]], %[[ADD_RES]]
  return %0, %2 : tensor<5xf32>, tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_addf_no_endomorphism_different_grid
func.func @all_reduce_arith_addf_no_endomorphism_different_grid(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: %[[ALL_REDUCE0:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG0]] on @grid0
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ALL_REDUCE1:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG1]] on @grid1
  %1 = shard.all_reduce %arg1 on @grid1 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ALL_REDUCE0]], %[[ALL_REDUCE1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: return %[[ADD_RES]]
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_addf_no_endomorphism_different_grid_axes
func.func @all_reduce_arith_addf_no_endomorphism_different_grid_axes(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: %[[ALL_REDUCE0:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG0]] on @grid0 grid_axes = [0]
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ALL_REDUCE1:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG1]] on @grid0 grid_axes = [1]
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [1]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ALL_REDUCE0]], %[[ALL_REDUCE1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: return %[[ADD_RES]]
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_addf_no_endomorphism_wrong_reduction_kind
func.func @all_reduce_arith_addf_no_endomorphism_wrong_reduction_kind(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  // CHECK: %[[ALL_REDUCE0:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG0]] on @grid0 grid_axes = [0] reduction = max
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0] reduction = max
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ALL_REDUCE1:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG1]] on @grid0 grid_axes = [0]
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ALL_REDUCE0]], %[[ALL_REDUCE1]]
  %2 = arith.addf %0, %1 : tensor<5xf32>
  // CHECK: return %[[ADD_RES]]
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_addf_no_endomorphism_different_operand_result_element_types
func.func @all_reduce_arith_addf_no_endomorphism_different_operand_result_element_types(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf64> {
  // CHECK: %[[ALL_REDUCE0:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG0]] on @grid0 grid_axes = [0]
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf64>
  // CHECK: %[[ALL_REDUCE1:[A-Za-z0-9_]*]] = shard.all_reduce %[[ARG1]] on @grid0 grid_axes = [0]
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0]
    : tensor<5xf32> -> tensor<5xf64>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.addf %[[ALL_REDUCE0]], %[[ALL_REDUCE1]]
  %2 = arith.addf %0, %1 : tensor<5xf64>
  // CHECK: return %[[ADD_RES]]
  return %2 : tensor<5xf64>
}

// Checks that `min(all_reduce(x), all_reduce(y))` gets transformed to
// `all_reduce(min(x, y))`.
// CHECK-LABEL: func.func @all_reduce_arith_minimumf_endomorphism
func.func @all_reduce_arith_minimumf_endomorphism(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg0: tensor<5xf32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xf32>
    %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0] reduction = min
    : tensor<5xf32> -> tensor<5xf32>
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0] reduction = min
    : tensor<5xf32> -> tensor<5xf32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.minimumf %[[ARG0]], %[[ARG1]]
  %2 = arith.minimumf %0, %1 : tensor<5xf32>
  // CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ADD_RES]] on @grid0 grid_axes = [0] reduction = min
  // CHECK: return %[[ALL_REDUCE_RES]]
  return %2 : tensor<5xf32>
}

// CHECK-LABEL: func.func @all_reduce_arith_minsi_endomorphism
func.func @all_reduce_arith_minsi_endomorphism(
    // CHECK-SAME: %[[ARG0:[A-Za-z0-9_]*]]: tensor<5xi32>
    %arg0: tensor<5xi32>,
    // CHECK-SAME: %[[ARG1:[A-Za-z0-9_]*]]: tensor<5xi32>
    %arg1: tensor<5xi32>) -> tensor<5xi32> {
  %0 = shard.all_reduce %arg0 on @grid0 grid_axes = [0] reduction = min
    : tensor<5xi32> -> tensor<5xi32>
  %1 = shard.all_reduce %arg1 on @grid0 grid_axes = [0] reduction = min
    : tensor<5xi32> -> tensor<5xi32>
  // CHECK: %[[ADD_RES:[A-Za-z0-9_]*]] = arith.minsi %[[ARG0]], %[[ARG1]]
  %2 = arith.minsi %0, %1 : tensor<5xi32>
  // CHECK: %[[ALL_REDUCE_RES:[A-Za-z0-9_]*]] = shard.all_reduce %[[ADD_RES]] on @grid0 grid_axes = [0] reduction = min
  // CHECK: return %[[ALL_REDUCE_RES]]
  return %2 : tensor<5xi32>
}

// Ensure this case without endomorphism op not crash.
// CHECK-LABEL: func.func @no_endomorphism_op
func.func @no_endomorphism_op(%arg0: tensor<2xi64>) -> i64 {
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  // CHECK: tensor.extract
  %extracted = tensor.extract %arg0[%c0] : tensor<2xi64>
  // CHECK: arith.maxsi
  %0 = arith.maxsi %extracted, %c1_i64 : i64
  return %0 : i64
}

// RUN: mlir-opt %s -canonicalize="test-convergence" -split-input-file | FileCheck %s

// CHECK-LABEL: func @fold_consecutive_scalar_mul_f32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant dense<6.000000e+00> : tensor<4x8xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[RESULT:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK: return %[[RESULT]]
func.func @fold_consecutive_scalar_mul_f32(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%arg0, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%mul1, %cst3 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func @fold_consecutive_scalar_mul_i32
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xi32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant dense<15> : tensor<4x8xi32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xi32>
// CHECK: %[[RESULT:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xi32>, tensor<4x8xi32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xi32>)
// CHECK: return %[[RESULT]]
func.func @fold_consecutive_scalar_mul_i32(%arg0: tensor<4x8xi32>) -> tensor<4x8xi32> {
  %cst5 = arith.constant dense<5> : tensor<4x8xi32>
  %cst3 = arith.constant dense<3> : tensor<4x8xi32>
  %empty = tensor.empty() : tensor<4x8xi32>
  %mul1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%arg0, %cst5 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  %mul2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%mul1, %cst3 : tensor<4x8xi32>, tensor<4x8xi32>)
    outs(%empty : tensor<4x8xi32>) -> tensor<4x8xi32>
  return %mul2 : tensor<4x8xi32>
}

// -----

// Scalar constant on the left-hand side.
// CHECK-LABEL: func @fold_scalar_mul_lhs
// CHECK-SAME: (%[[ARG:.*]]: tensor<4x8xf32>)
// CHECK-DAG: %[[COMBINED:.*]] = arith.constant dense<1.200000e+01> : tensor<4x8xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK: %[[RESULT:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK-SAME: ins(%[[ARG]], %[[COMBINED]] : tensor<4x8xf32>, tensor<4x8xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<4x8xf32>)
// CHECK: return %[[RESULT]]
func.func @fold_scalar_mul_lhs(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst4 = arith.constant dense<4.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%cst4, %arg0 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%cst3, %mul1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

// -----

// Do not fold when the inner mul has multiple uses.
// CHECK-LABEL: func @no_fold_multi_use
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
func.func @no_fold_multi_use(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %cst2 = arith.constant dense<2.0> : tensor<4x8xf32>
  %cst3 = arith.constant dense<3.0> : tensor<4x8xf32>
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%arg0, %cst2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%mul1, %cst3 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  // Extra use of mul1 prevents folding.
  %add = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%mul2, %mul1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %add : tensor<4x8xf32>
}

// -----

// Do not fold when neither operand is a scalar constant.
// CHECK-LABEL: func @no_fold_non_const
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
func.func @no_fold_non_const(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>,
                             %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %empty = tensor.empty() : tensor<4x8xf32>
  %mul1 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  %mul2 = linalg.elementwise kind=#linalg.elementwise_kind<mul>
    ins(%mul1, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>)
    outs(%empty : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %mul2 : tensor<4x8xf32>
}

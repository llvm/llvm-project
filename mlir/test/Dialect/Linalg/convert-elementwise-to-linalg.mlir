// RUN: mlir-opt -convert-elementwise-to-linalg -split-input-file %s | FileCheck %s

// In-depth checking of the linalg.generic op for a very trivial case.
// CHECK: #[[$MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @addf_rank0
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<f32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<f32>
func @addf_rank0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: %{{.*}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]]
  // CHECK-SAME: iterator_types = []
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
  // CHECK-SAME: outs(%[[ARG0]]
  //      CHECK: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
  //      CHECK:   %[[YIELD:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
  //      CHECK:   linalg.yield %[[YIELD]] : f32
  //      CHECK: } -> tensor<f32>
  %0 = arith.addf %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Check indexing maps and iterator types for the rank > 0 case.
// CHECK-LABEL: func @addf_rank1
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<?xf32>
func @addf_rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
  // CHECK-SAME: outs(%[[ARG0]]
  %0 = arith.addf %arg0, %arg1 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Check a unary op.
// CHECK-LABEL: func @exp
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<f32>
func @exp(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]]
  // CHECK-SAME: outs(%[[ARG0]]
  // CHECK: ^bb0(%[[SCALAR:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[YIELD:.*]] = math.exp %[[SCALAR]] : f32
  // CHECK:   linalg.yield %[[YIELD]] : f32
  %0 = math.exp %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----

// Check a case with varying operand types.
// CHECK-LABEL: func @select
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<i1>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<i32>
//  CHECK-SAME:   %[[ARG2:[0-9a-zA-Z]*]]: tensor<i32>
func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK-SAME: outs(%[[ARG1]]
  // CHECK: ^bb0(%[[PRED:.*]]: i1, %[[TRUE_VAL:.*]]: i32, %[[FALSE_VAL:.*]]: i32, %{{.*}}: i32):
  // CHECK:   select %[[PRED]], %[[TRUE_VAL]], %[[FALSE_VAL]] : i32
  %0 = select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Spot-check an op that requires copying attributes properly to the created scalar op.
// Also checks proper init_tensor usage.
// CHECK-LABEL: func @cmpf(
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<f32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<f32>
func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  // CHECK: %[[INIT:.*]] = linalg.init_tensor [] : tensor<i1>
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: i1):
  // CHECK: arith.cmpf olt, %{{.*}}, %{{.*}} : f32
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<f32>
  return %0 : tensor<i1>
}

// -----

// Check proper init_tensor usage in a mixed case.
// CHECK-LABEL: func @cmpf(
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<4x?x?x8x2x?xf32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<4x?x?x8x2x?xf32>
func @cmpf(%arg0: tensor<4x?x?x8x2x?xf32>, %arg1: tensor<4x?x?x8x2x?xf32>) -> tensor<4x?x?x8x2x?xi1> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[D2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: %[[D5:.*]] = tensor.dim %[[ARG0]], %[[C5]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[INIT:.*]] = linalg.init_tensor [4, %[[D1]], %[[D2]], 8, 2, %[[D5]]] : tensor<4x?x?x8x2x?xi1>
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: i1):
  // CHECK: arith.cmpf olt, %{{.*}}, %{{.*}} : f32
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<4x?x?x8x2x?xf32>
  return %0 : tensor<4x?x?x8x2x?xi1>
}


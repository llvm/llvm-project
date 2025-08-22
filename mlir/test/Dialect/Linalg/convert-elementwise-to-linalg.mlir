// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-elementwise-to-linalg))" -split-input-file %s | FileCheck %s

// In-depth checking of the linalg.generic op for a very trivial case.
// CHECK: #[[$MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @addf_rank0
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<f32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<f32>
func.func @addf_rank0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
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

// Test a binary elementwise op with a tensor and a scalar operand.
// CHECK-LABEL: func @addf_tensor_plus_scalar_rank1
//  CHECK-SAME:   %[[T:[0-9a-zA-Z]*]]: tensor<?xf32>, %[[S:[0-9a-zA-Z]*]]: f32
func.func @addf_tensor_plus_scalar_rank1(%t: tensor<?xf32>, %s: f32) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %t, %c0 : tensor<?xf32>
  %init = tensor.empty(%d0) : tensor<?xf32>
  %splat = linalg.fill ins(%s : f32) outs(%init : tensor<?xf32>) -> tensor<?xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK-SAME: ins(%[[T]], %{{.*}}
  %0 = arith.addf %t, %splat : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// Test a comparison op between a tensor and a scalar.
// CHECK-LABEL: func @cmpf_tensor_scalar
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]*]]: tensor<?xf32>, %[[S:[0-9a-zA-Z]*]]: f32
func.func @cmpf_tensor_scalar(%a: tensor<?xf32>, %s: f32) -> tensor<?xi1> {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %a, %c0 : tensor<?xf32>
  %initS = tensor.empty(%d0) : tensor<?xf32>
  %splat = linalg.fill ins(%s : f32) outs(%initS : tensor<?xf32>) -> tensor<?xf32>

  %init = tensor.empty(%d0) : tensor<?xi1>
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[A]], %{{.*}}
  %0 = arith.cmpf olt, %a, %splat : tensor<?xf32>
  return %0 : tensor<?xi1>
}

// Test a binary elementwise op with a tensor and a zero-dimensional
// (rank-0) tensor.
// CHECK-LABEL: func @addf_tensor_plus_rank0_tensor
//  CHECK-SAME:   %[[T:[0-9a-zA-Z]*]]: tensor<4xf32>, %[[R0:[0-9a-zA-Z]*]]: tensor<f32>
func.func @addf_tensor_plus_rank0_tensor(%t: tensor<4xf32>, %r0: tensor<f32>) -> tensor<4xf32> {
  %c = tensor.extract %r0[] : tensor<f32>
  %init = tensor.empty() : tensor<4xf32>
  %splat = linalg.fill ins(%c : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[T]], %{{.*}}
  %0 = arith.addf %t, %splat : tensor<4xf32>
  return %0 : tensor<4xf32>
}


// -----

// Check indexing maps and iterator types for the rank > 0 case.
// CHECK-LABEL: func @addf_rank1
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<?xf32>
func.func @addf_rank1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
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
func.func @exp(%arg0: tensor<f32>) -> tensor<f32> {
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
func.func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]], %[[ARG2]]
  // CHECK-SAME: outs(%[[ARG1]]
  // CHECK: ^bb0(%[[PRED:.*]]: i1, %[[TRUE_VAL:.*]]: i32, %[[FALSE_VAL:.*]]: i32, %{{.*}}: i32):
  // CHECK:   arith.select %[[PRED]], %[[TRUE_VAL]], %[[FALSE_VAL]] : i32
  %0 = arith.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
  return %0 : tensor<i32>
}

// -----

// Spot-check an op that requires copying attributes properly to the created scalar op.
// Also checks proper init_tensor usage.
// CHECK-LABEL: func @cmpf(
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]*]]: tensor<f32>
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]*]]: tensor<f32>
func.func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  // CHECK: %[[INIT:.*]] = tensor.empty() : tensor<i1>
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
func.func @cmpf(%arg0: tensor<4x?x?x8x2x?xf32>, %arg1: tensor<4x?x?x8x2x?xf32>) -> tensor<4x?x?x8x2x?xi1> {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[D2:.*]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[C5:.*]] = arith.constant 5 : index
  // CHECK: %[[D5:.*]] = tensor.dim %[[ARG0]], %[[C5]] : tensor<4x?x?x8x2x?xf32>
  // CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D2]], %[[D5]]) : tensor<4x?x?x8x2x?xi1>
  // CHECK: linalg.generic
  // CHECK-SAME:  ins(%[[ARG0]], %[[ARG1]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: i1):
  // CHECK: arith.cmpf olt, %{{.*}}, %{{.*}} : f32
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<4x?x?x8x2x?xf32>
  return %0 : tensor<4x?x?x8x2x?xi1>
}


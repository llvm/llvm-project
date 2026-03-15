// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(resolve-shaped-type-result-dims{error-on-pattern-iteration-limit=false}))" -split-input-file | FileCheck %s
// See %test_unreifiable_result_shape below for why `error-on-partition-iteration-limit` is set to false.

func.func @result_shape(%arg0 : tensor<2x3x?xf32>, %arg1 : tensor<?x5xf32>)
    -> (index, index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = "test.op_with_result_shape_interface"(%arg0, %arg1)
      : (tensor<2x3x?xf32>, tensor<?x5xf32>) -> (tensor<?x5xf32>, tensor<2x3x?xf32>)
  %1 = tensor.dim %0#0, %c0 : tensor<?x5xf32>
  %2 = tensor.dim %0#0, %c1 : tensor<?x5xf32>
  %3 = tensor.dim %0#1, %c0 : tensor<2x3x?xf32>
  %4 = tensor.dim %0#1, %c1 : tensor<2x3x?xf32>
  %5 = tensor.dim %0#1, %c2 : tensor<2x3x?xf32>
  return %1, %2, %3, %4, %5 : index, index, index, index, index
}
// CHECK-LABEL: func @result_shape(
//  CHECK-SAME:   %[[ARG_0:[a-z0-9]*]]: tensor<2x3x?xf32>
//  CHECK-SAME:   %[[ARG_1:[a-z0-9]*]]: tensor<?x5xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG_1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG_0]], %[[C2]]
//       CHECK:   return %[[D0]], %[[C5]], %[[C2]], %[[C3]], %[[D1]]

// -----

// Test result shape reification for an operation that implements only 
// `reifyResultShapes` method of the `InferShapedTypeOpInterface`.
func.func @reify_shaped_type_using_reify_result_shapes(%arg0 : tensor<2x3x?xf32>, %arg1 : tensor<?x5xf32>)
    -> (index, index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = "test.reify_shaped_type_using_reify_result_shapes"(%arg0, %arg1)
      : (tensor<2x3x?xf32>, tensor<?x5xf32>) -> (tensor<?x5xf32>, tensor<2x3x?xf32>)
  %1 = tensor.dim %0#0, %c0 : tensor<?x5xf32>
  %2 = tensor.dim %0#0, %c1 : tensor<?x5xf32>
  %3 = tensor.dim %0#1, %c0 : tensor<2x3x?xf32>
  %4 = tensor.dim %0#1, %c1 : tensor<2x3x?xf32>
  %5 = tensor.dim %0#1, %c2 : tensor<2x3x?xf32>
  return %1, %2, %3, %4, %5 : index, index, index, index, index
}
// CHECK-LABEL: func @reify_shaped_type_using_reify_result_shapes(
//  CHECK-SAME:   %[[ARG_0:[a-z0-9]*]]: tensor<2x3x?xf32>
//  CHECK-SAME:   %[[ARG_1:[a-z0-9]*]]: tensor<?x5xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG_1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG_0]], %[[C2]]
//       CHECK:   return %[[D0]], %[[C5]], %[[C2]], %[[C3]], %[[D1]]

// -----

// Test result shape reification for an operation that implements only 
// `reifyShapeOfResult` method of the `InferShapedTypeOpInterface`.
func.func @reify_shaped_type_using_reify_shape_of_result(%arg0 : tensor<2x3x?xf32>, %arg1 : tensor<?x5xf32>)
    -> (index, index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = "test.reify_shaped_type_using_reify_result_shapes"(%arg0, %arg1)
      : (tensor<2x3x?xf32>, tensor<?x5xf32>) -> (tensor<?x5xf32>, tensor<2x3x?xf32>)
  %1 = tensor.dim %0#0, %c0 : tensor<?x5xf32>
  %2 = tensor.dim %0#0, %c1 : tensor<?x5xf32>
  %3 = tensor.dim %0#1, %c0 : tensor<2x3x?xf32>
  %4 = tensor.dim %0#1, %c1 : tensor<2x3x?xf32>
  %5 = tensor.dim %0#1, %c2 : tensor<2x3x?xf32>
  return %1, %2, %3, %4, %5 : index, index, index, index, index
}
// CHECK-LABEL: func @reify_shaped_type_using_reify_shape_of_result(
//  CHECK-SAME:   %[[ARG_0:[a-z0-9]*]]: tensor<2x3x?xf32>
//  CHECK-SAME:   %[[ARG_1:[a-z0-9]*]]: tensor<?x5xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG_1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG_0]], %[[C2]]
//       CHECK:   return %[[D0]], %[[C5]], %[[C2]], %[[C3]], %[[D1]]

// -----

// Test result shape reification for an operation that implements only 
// `reifyDimOfResult` method of the `InferShapedTypeOpInterface`.
func.func @reify_shaped_type_using_reify_dim_of_result(%arg0 : tensor<2x3x?xf32>, %arg1 : tensor<?x5xf32>)
    -> (index, index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0:2 = "test.reify_shaped_type_using_reify_result_shapes"(%arg0, %arg1)
      : (tensor<2x3x?xf32>, tensor<?x5xf32>) -> (tensor<?x5xf32>, tensor<2x3x?xf32>)
  %1 = tensor.dim %0#0, %c0 : tensor<?x5xf32>
  %2 = tensor.dim %0#0, %c1 : tensor<?x5xf32>
  %3 = tensor.dim %0#1, %c0 : tensor<2x3x?xf32>
  %4 = tensor.dim %0#1, %c1 : tensor<2x3x?xf32>
  %5 = tensor.dim %0#1, %c2 : tensor<2x3x?xf32>
  return %1, %2, %3, %4, %5 : index, index, index, index, index
}
// CHECK-LABEL: func @reify_shaped_type_using_reify_dim_of_result(
//  CHECK-SAME:   %[[ARG_0:[a-z0-9]*]]: tensor<2x3x?xf32>
//  CHECK-SAME:   %[[ARG_1:[a-z0-9]*]]: tensor<?x5xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG_1]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG_0]], %[[C2]]
//       CHECK:   return %[[D0]], %[[C5]], %[[C2]], %[[C3]], %[[D1]]

// -----

// This tests also indicates a problem with the approach of just using `reifyShapes`
// without being specific about {result, dim} that needs to be resolved. The
// `reifyShapes` implementations introduces `dim` operations that are effectively
// dead, but it creates an infinite loop on pattern application (which eventually
// bails on hitting the iteration limit). This is the pitfall of this legacy
// mechanism.

func.func @test_unreifiable_result_shapes(%arg0 : tensor<?x?xf32>)
    -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "test.unreifiable_result_shapes"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}
// CHECK-LABEL: func @test_unreifiable_result_shapes(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[OP:.+]] = "test.unreifiable_result_shapes"(%[[ARG0]])
//       CHECK:   %[[D1:.+]] = tensor.dim %[[OP]], %[[C1]]
//       CHECK:   return %[[D0]], %[[D1]]
// -----

func.func @test_unreifiable_result_shape(%arg0 : tensor<?x?xf32>)
    -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "test.unreifiable_result_shape"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}
// CHECK-LABEL: func @test_unreifiable_result_shape(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[OP:.+]] = "test.unreifiable_result_shape"(%[[ARG0]])
//       CHECK:   %[[D1:.+]] = tensor.dim %[[OP]], %[[C1]]
//       CHECK:   return %[[D0]], %[[D1]]

// -----

func.func @test_unreifiable_dim_of_result_shape(%arg0 : tensor<?x?xf32>)
    -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "test.unreifiable_dim_of_result_shape"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %d0, %d1 : index, index
}
// CHECK-LABEL: func @test_unreifiable_dim_of_result_shape(
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[OP:.+]] = "test.unreifiable_dim_of_result_shape"(%[[ARG0]])
//       CHECK:   %[[D1:.+]] = tensor.dim %[[OP]], %[[C1]]
//       CHECK:   return %[[D0]], %[[D1]]

// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-bubble-up-extract-slice-through-tiling-interface %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @bubble_single_slice(
//   CHECK-DAG:   %[[SLICE0:.*]] = tensor.extract_slice %arg0[4, 8] [4, 4] [1, 1]
//   CHECK-DAG:   %[[SLICE1:.*]] = tensor.extract_slice %arg1[8] [4] [1]
//   CHECK-DAG:   %[[SLICE2:.*]] = tensor.extract_slice %arg0[4, 8] [4, 4] [1, 1]
//       CHECK:   %[[GENERIC:.*]] = linalg.generic
//  CHECK-SAME:       ins(%[[SLICE0]], %[[SLICE1]] : tensor<4x4xf32>, tensor<4xf32>)
//  CHECK-SAME:       outs(%[[SLICE2]] : tensor<4x4xf32>)
//       CHECK:   return %[[GENERIC]]
func.func @bubble_single_slice(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>) -> tensor<4x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16xf32>)
      outs(%arg0 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
  } -> tensor<16x16xf32>
  %1 = tensor.extract_slice %0[4, 8] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @bubble_multiple_non_overlapping_slices(
//       CHECK:   %[[S0:.*]] = tensor.extract_slice %arg0[0, 0] [4, 8] [1, 1]
//       CHECK:   %[[S1:.*]] = tensor.extract_slice %arg1[0] [8] [1]
//       CHECK:   %[[S2:.*]] = tensor.extract_slice %arg0[0, 0] [4, 8] [1, 1]
//       CHECK:   %[[GENERIC1:.*]] = linalg.generic
//  CHECK-SAME:       ins(%[[S0]], %[[S1]] : tensor<4x8xf32>, tensor<8xf32>)
//  CHECK-SAME:       outs(%[[S2]] : tensor<4x8xf32>)
//       CHECK:   %[[S3:.*]] = tensor.extract_slice %arg0[0, 8] [4, 8] [1, 1]
//       CHECK:   %[[S4:.*]] = tensor.extract_slice %arg1[8] [8] [1]
//       CHECK:   %[[S5:.*]] = tensor.extract_slice %arg0[0, 8] [4, 8] [1, 1]
//       CHECK:   %[[GENERIC2:.*]] = linalg.generic
//  CHECK-SAME:       ins(%[[S3]], %[[S4]] : tensor<4x8xf32>, tensor<8xf32>)
//  CHECK-SAME:       outs(%[[S5]] : tensor<4x8xf32>)
//       CHECK:   return %[[GENERIC1]], %[[GENERIC2]]
func.func @bubble_multiple_non_overlapping_slices(%arg0: tensor<16x16xf32>, %arg1: tensor<16xf32>)
    -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16xf32>)
      outs(%arg0 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
  } -> tensor<16x16xf32>
  %1 = tensor.extract_slice %0[0, 0] [4, 8] [1, 1] : tensor<16x16xf32> to tensor<4x8xf32>
  %2 = tensor.extract_slice %0[0, 8] [4, 8] [1, 1] : tensor<16x16xf32> to tensor<4x8xf32>
  return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// -----

// CHECK-LABEL: func.func @bubble_through_matmul(
//   CHECK-DAG:   %[[LHS_SLICE:.*]] = tensor.extract_slice %arg0[2, 0] [4, 8] [1, 1]
//   CHECK-DAG:   %[[RHS_SLICE:.*]] = tensor.extract_slice %arg1[0, 2] [8, 4] [1, 1]
//   CHECK-DAG:   %[[DST_SLICE:.*]] = tensor.extract_slice %arg2[2, 2] [4, 4] [1, 1]
//       CHECK:   %[[MATMUL:.*]] = linalg.matmul ins(%[[LHS_SLICE]], %[[RHS_SLICE]] :
//       CHECK:   return %[[MATMUL]]
func.func @bubble_through_matmul(%lhs: tensor<8x8xf32>, %rhs: tensor<8x8xf32>,
                                 %dst: tensor<8x8xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%lhs, %rhs : tensor<8x8xf32>, tensor<8x8xf32>)
                     outs(%dst : tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = tensor.extract_slice %0[2, 2] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @bubble_through_fill(
//       CHECK:   %[[CST:.*]] = arith.constant 1.000000e+00 : f32
//       CHECK:   %[[EMPTY:.*]] = tensor.empty()
//       CHECK:   %[[SLICE:.*]] = tensor.extract_slice %[[EMPTY]][4, 4] [4, 4] [1, 1]
//       CHECK:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[SLICE]] : tensor<4x4xf32>)
//       CHECK:   return %[[FILL]]
func.func @bubble_through_fill() -> tensor<4x4xf32> {
  %cst = arith.constant 1.0 : f32
  %empty = tensor.empty() : tensor<16x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
  %slice = tensor.extract_slice %fill[4, 4] [4, 4] [1, 1] : tensor<16x16xf32> to tensor<4x4xf32>
  return %slice : tensor<4x4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @bubble_with_dynamic_dims(
//       CHECK:   %[[SLICE0:.*]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1]
//       CHECK:   %[[SLICE1:.*]] = tensor.extract_slice %arg1[%arg3] [%arg5] [1]
//       CHECK:   %[[SLICE2:.*]] = tensor.extract_slice %arg0[%arg2, %arg3] [%arg4, %arg5] [1, 1]
//       CHECK:   %[[GENERIC:.*]] = linalg.generic
//  CHECK-SAME:       ins(%[[SLICE0]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?xf32>)
//  CHECK-SAME:       outs(%[[SLICE2]] : tensor<?x?xf32>)
//       CHECK:   return %[[GENERIC]]
func.func @bubble_with_dynamic_dims(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>,
                                    %off0: index, %off1: index,
                                    %sz0: index, %sz1: index) -> tensor<?x?xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
      outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.extract_slice %0[%off0, %off1] [%sz0, %sz1] [1, 1]
      : tensor<?x?xf32> to tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

/// Negative test: result has non-extract_slice consumer.

// CHECK-LABEL: func.func @no_bubble_non_slice_consumer(
//       CHECK:   %[[GENERIC:.*]] = linalg.generic
//  CHECK-SAME:       ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16xf32>)
//  CHECK-SAME:       outs(%arg0 : tensor<16x16xf32>)
//       CHECK:   return %[[GENERIC]] : tensor<16x16xf32>
func.func @no_bubble_non_slice_consumer(%arg0: tensor<16x16xf32>,
                                        %arg1: tensor<16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16xf32>)
      outs(%arg0 : tensor<16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
  } -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

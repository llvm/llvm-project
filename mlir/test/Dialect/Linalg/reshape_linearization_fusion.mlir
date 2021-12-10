// RUN: mlir-opt -split-input-file -linalg-fold-reshape-ops-by-linearization %s | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?xi32>)
  -> tensor<?x?x4x?xi32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2], [3]] :
    tensor<?x?x?xi32> into tensor<?x?x4x?xi32>
  %1 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%0 : tensor<?x?x4x?xi32>)
    outs(%0 : tensor<?x?x4x?xi32>) {
  ^bb0(%arg6: i32, %arg7 : i32):       // no predecessors
    %idx = linalg.index 0 : index
    %2 = arith.index_cast %idx : index to i32
    %3 = arith.addi %arg6, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<?x?x4x?xi32>
  return %1 : tensor<?x?x4x?xi32>
}
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
//   CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//       CHECK: func @generic_op_reshape_producer_fusion
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?xi32>
//       CHECK:   %[[T0:.+]] = tensor.expand_shape %[[ARG0]]
//  CHECK-SAME:     [0], [1, 2], [3]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP3]], #[[MAP4]]]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<?x?x?xi32>)
//  CHECK-SAME:     outs(%[[T0]] : tensor<?x?x4x?xi32>)
//       CHECK:   %[[IDX:.+]] = linalg.index 0 : index
//  CHECK-NEXT:   %[[IDX_CASTED:.+]] = arith.index_cast %[[IDX]] : index to i32

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?x4x5xi32>)
  -> tensor<?x?xi32> {
  %0 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%arg0 : tensor<?x?x4x5xi32>) outs(%arg0 : tensor<?x?x4x5xi32>) {
  ^bb0(%arg6: i32, %arg7: i32):       // no predecessors
    %idx = linalg.index 0 : index
    %2 = arith.index_cast %idx : index to i32
    %3 = arith.addi %arg6, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<?x?x4x5xi32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3]] :
    tensor<?x?x4x5xi32> into tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>
//       CHECK: func @generic_op_reshape_consumer_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x4x5xi32>
//       CHECK:   %[[T0:.+]] = tensor.collapse_shape %[[ARG0]]
//  CHECK-SAME:     [0], [1, 2, 3]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
//  CHECK-SAME:     outs(%[[T0]] : tensor<?x?xi32>)
//       CHECK:   %[[IDX:.+]] = linalg.index 0 : index
//  CHECK-NEXT:   %[[IDX_CASTED:.+]] = arith.index_cast %[[IDX]] : index to i32
//   CHECK-NOT:   tensor.collapse_shape

// -----

#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_021_permultation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<3x7x5xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]]
      : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.init_tensor [3, 7, 5] : tensor<3x7x5xf32>
  %2 = linalg.generic
    {indexing_maps = [#map2, #map3],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : tensor<3x5x7xf32>) outs(%1 : tensor<3x7x5xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<3x7x5xf32>
    return %2 : tensor<3x7x5xf32>
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 + d2 * 7)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//       CHECK: func @generic_op_021_permultation_reshape_producer_fusion
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]

// -----

#map2 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
func @generic_op_120_permutation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<5x7x3xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]]
      : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.init_tensor [5, 7, 3] : tensor<5x7x3xf32>
  %2 = linalg.generic
    {indexing_maps = [#map2, #map3],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : tensor<3x5x7xf32>) outs(%1 : tensor<5x7x3xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<5x7x3xf32>
    return %2 : tensor<5x7x3xf32>
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0 * 7 + d2)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//       CHECK: func @generic_op_120_permutation_reshape_producer_fusion
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_102_permultation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<5x3x7xf32> {
  %0 = tensor.expand_shape %arg0 [[0], [1, 2]]
      : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.init_tensor [5, 3, 7] : tensor<5x3x7xf32>
  %2 = linalg.generic
    {indexing_maps = [#map2, #map3],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : tensor<3x5x7xf32>) outs(%1 : tensor<5x3x7xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<5x3x7xf32>
    return %2 : tensor<5x3x7xf32>
}


//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0 * 7 + d2)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//       CHECK: func @generic_op_102_permultation_reshape_producer_fusion
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func @generic_op_102_permultation_reshape_consumer_fusion(%arg0 : tensor<3x5x7xf32>) -> tensor<5x21xf32> {
  %0 = linalg.init_tensor [5, 3, 7] : tensor<5x3x7xf32>
  %1 = linalg.generic
    {indexing_maps = [#map0, #map1],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<3x5x7xf32>) outs(%0 : tensor<5x3x7xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):  // no predecessors
      linalg.yield %arg2 : f32
  } -> tensor<5x3x7xf32>
  %2 = tensor.collapse_shape %1 [[0], [1, 2]]
      : tensor<5x3x7xf32> into tensor<5x21xf32>
  return %2 : tensor<5x21xf32>
}
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d1, d0 * 7 + d2)>
//       CHECK: func @generic_op_102_permultation_reshape_consumer_fusion
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<3x5x7xf32>
//       CHECK:   %[[T0:.+]] = linalg.init_tensor [5, 3, 7]
//       CHECK:   %[[T1:.+]] = tensor.collapse_shape %[[T0]]
//  CHECK-SAME:     [0], [1, 2]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<3x5x7xf32>)
//  CHECK-SAME:     outs(%[[T1]] : tensor<5x21xf32>)

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_nofusion(%arg0 : tensor<?x?x?x5xf32>,
                                           %arg1 : tensor<?x?x?x5xf32>) ->
                                           tensor<?x?xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?x?x5xf32>, tensor<?x?x?x5xf32>)
      outs(%arg0 : tensor<?x?x?x5xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       // no predecessors
      %1 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x?x5xf32>
  %1 = tensor.collapse_shape %0 [[0], [1, 2, 3]] :
    tensor<?x?x?x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// CHECK-LABEL: func @generic_op_reshape_consumer_nofusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x5xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?x5xf32>
//       CHECK:   %[[NOFUSE:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]]
//       CHECK:   %[[RESULT:.+]] = tensor.collapse_shape %[[NOFUSE]]
//       CHECK:   return %[[RESULT]]


// -----

func @generic_op_permultation_reshape_consumer_fusion_unused_dim(%arg0 : tensor<6x1xf32>) -> tensor<6xi32> {
  %0 = linalg.init_tensor [6, 1] : tensor<6x1xi32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
   iterator_types = ["parallel", "parallel"]}
   ins(%arg0 : tensor<6x1xf32>) outs(%0 : tensor<6x1xi32>) {
    ^bb0(%arg3: f32, %arg4: i32):  // no predecessors
      %5 = arith.fptosi %arg3 : f32 to i32
      linalg.yield %5 : i32
    } -> tensor<6x1xi32>
    %6 = tensor.collapse_shape %1 [[0, 1]] : tensor<6x1xi32> into tensor<6xi32>
  return %6 : tensor<6xi32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>
//       CHECK: func @generic_op_permultation_reshape_consumer_fusion_unused_dim
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<6x1xf32>
//       CHECK:   %[[T0:.+]] = linalg.init_tensor [6, 1]
//       CHECK:   %[[T1:.+]] = tensor.collapse_shape %[[T0]]
//  CHECK-SAME:   [0, 1]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<6x1xf32>)
//  CHECK-SAME:     outs(%[[T1]] : tensor<6xi32>)

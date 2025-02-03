// RUN: mlir-opt %s -test-linalg-elementwise-fusion-patterns=fuse-with-reshape-by-expansion -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map2 = affine_map<(d0, d1, d2) -> ()>
func.func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x4x?xf32>,
                                         %arg1 : tensor<?x?x?xf32>,
                                         %arg2 : f32) ->
                                         tensor<?x?x?xf32>
{
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] :
    tensor<?x?x4x?xf32> into tensor<?x?x?xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map1, #map2, #map1],
     iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%0, %arg1, %arg2 : tensor<?x?x?xf32>, tensor<?x?x?xf32>, f32)
       outs(%arg1 : tensor<?x?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %s: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
  } -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0, d1, d2, d3) -> ()>
//      CHECK: func @generic_op_reshape_producer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x4x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: f32
//      CHECK:   %[[C4:.+]] = arith.constant 4 : index
//      CHECK:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<?x?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM_1]], %[[C4]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0], [1], [2, 3]] output_shape [%[[DIM]], %[[DIM_0]], %[[VAL_0]], 4] : tensor<?x?x?xf32> into tensor<?x?x?x4xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_3:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_4:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<?x?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_4]], %[[C4]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0], [1], [2, 3]] output_shape [%[[DIM_2]], %[[DIM_3]], %[[VAL_1]], 4] : tensor<?x?x?xf32> into tensor<?x?x?x4xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP5]], #[[MAP6]], #[[MAP7]], #[[MAP6]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T1]], %[[ARG2]] : tensor<?x?x4x?xf32>, tensor<?x?x?x4xf32>, f32)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x?x?x4xf32>)
//      CHECK:   %[[T4:.+]] = tensor.collapse_shape %[[T3]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     tensor<?x?x?x4xf32> into tensor<?x?x?xf32>
//      CHECK:   return %[[T4]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
func.func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?xf32>,
                                         %arg1 : tensor<?x?xf32>,
                                         %arg2 : f32,
                                         %sz0: index,
                                         %sz1: index) ->
                                         tensor<?x4x?x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map1, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>, f32)
       outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %s: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2, 3]] output_shape [%sz0, 4, %sz1, 5] :
    tensor<?x?xf32> into tensor<?x4x?x5xf32>
  return %1 : tensor<?x4x?x5xf32>
}

//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> ()>

//      CHECK: func @generic_op_reshape_consumer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: f32
// CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index
//      CHECK:   %[[C20:.+]] = arith.constant 20 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM_0]], %[[C20]] : index
//      CHECK:   %[[T0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM]], 4, %[[VAL_0]], 5] : tensor<?x?xf32> into tensor<?x4x?x5xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_2]], %[[C20]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_1]], 4, %[[VAL_1]], 5] : tensor<?x?xf32> into tensor<?x4x?x5xf32>
//      CHECK:   %[[DIM_4:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_5:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_5]], %[[C20]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_4]], 4, %[[VAL_2]], 5] : tensor<?x?xf32> into tensor<?x4x?x5xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP3]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]], %[[ARG2]] : tensor<?x4x?x5xf32>, tensor<?x4x?x5xf32>, f32)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x4x?x5xf32>)
//      CHECK:   return %[[T3]] : tensor<?x4x?x5xf32>


// -----

func.func @reshape_as_consumer_permutation
  (%a : tensor<?x?x?xf32>, %b : tensor<?x?xf32>, %sz0: index, %sz1: index, %sz2: index)
    -> tensor<?x2x?x3x4x?xf32> {
  %c = linalg.generic {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
         iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?xf32>)
         outs(%a : tensor<?x?x?xf32>) {
       ^bb0(%arg0 : f32, %arg1: f32, %s: f32):
         %1 = arith.addf %arg0, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?x?xf32>
  %d = tensor.expand_shape %c [[0, 1], [2], [3, 4, 5]] output_shape [%sz0, 2, %sz1, 3, 4, %sz2] : tensor<?x?x?xf32> into tensor<?x2x?x3x4x?xf32>
  return %d : tensor<?x2x?x3x4x?xf32>
}
//  CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//  CHECK-DAG: #[[MAP9:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//  CHECK-DAG: #[[MAP10:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//      CHECK: func @reshape_as_consumer_permutation
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index, %[[SZ2:.+]]: index
//      CHECK:   %[[C12:.+]] = arith.constant 12 : index
//      CHECK:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C12]] : index
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_0]], %[[C2]] : index
//      CHECK:   %[[T0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3, 4], [5]] output_shape [3, 4, %[[VAL_0]], %[[VAL_1]], 2, %[[DIM_1]]] : tensor<?x?x?xf32> into tensor<3x4x?x?x2x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_3:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_2]], %[[C12]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3]] output_shape [3, 4, %[[VAL_2]], %[[DIM_3]]] : tensor<?x?xf32> into tensor<3x4x?x?xf32>
//      CHECK:   %[[DIM_5:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_6:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
//      CHECK:   %[[DIM_7:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
//      CHECK:   %[[VAL_3:.+]] = arith.divsi %[[DIM_5]], %[[C2]] : index
//      CHECK:   %[[VAL_4:.+]] = arith.divsi %[[DIM_7]], %[[C12]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2], [3, 4, 5]] output_shape [%[[VAL_3]], 2, %[[DIM_6]], 3, 4, %[[VAL_4]]] : tensor<?x?x?xf32> into tensor<?x2x?x3x4x?xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP8]], #[[MAP9]], #[[MAP10]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<3x4x?x?x2x?xf32>, tensor<3x4x?x?xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x2x?x3x4x?xf32>)
//      CHECK:   return %[[T3]] : tensor<?x2x?x3x4x?xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

func.func @generic_op_reshape_consumer_static(%arg0: tensor<264x4xf32>)
                                            -> tensor<8x33x4xf32> {
  %cst = arith.constant dense<2.000000e+00> : tensor<264x4xf32>
  %0 = tensor.empty() : tensor<264x4xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %cst : tensor<264x4xf32>, tensor<264x4xf32>)
       outs(%0 : tensor<264x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %s: f32):
      %2 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<264x4xf32>
  %2 = tensor.expand_shape %1 [[0, 1], [2]] output_shape [8, 33, 4] :
    tensor<264x4xf32> into tensor<8x33x4xf32>
  return %2 : tensor<8x33x4xf32>
}

//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @generic_op_reshape_consumer_static
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<264x4xf32>
//  CHECK-DAG:   %[[CST:.+]] = arith.constant
// CHECK-SAME:     : tensor<8x33x4xf32>
//  CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//      CHECK:   %[[T0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2]] output_shape [8, 33, 4] : tensor<264x4xf32> into tensor<8x33x4xf32>
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2]] output_shape [8, 33, 4] : tensor<264x4xf32> into tensor<8x33x4xf32>
//      CHECK:   %[[T2:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[CST]] :
// CHECK-SAME:     outs(%[[T1]] : tensor<8x33x4xf32>)
//      CHECK:   return %[[T2]] : tensor<8x33x4xf32>

// -----

#map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func.func @indexed_consumer_reshape_producer_fusion(%arg0 : tensor<?x?x4x?xi32>,
                                         %arg1 : tensor<?x?x?xi32>) ->
                                         tensor<?x?x?xi32>
{
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2], [3]]:
    tensor<?x?x4x?xi32> into tensor<?x?x?xi32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map1, #map1],
     iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
      outs(%0 : tensor<?x?x?xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %s: i32):
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %idx2 = linalg.index 2 : index
      %1 = arith.muli %arg3, %arg4 : i32
      %2 = arith.index_cast %idx0 : index to i32
      %3 = arith.addi %1, %2 : i32
      %4 = arith.index_cast %idx1 : index to i32
      %5 = arith.addi %3, %4 : i32
      %6 = arith.index_cast %idx2 : index to i32
      %7 = arith.addi %5, %6 : i32
      linalg.yield %7 : i32
  } -> tensor<?x?x?xi32>
  return %1 : tensor<?x?x?xi32>
}

// Only check the body in the indexed version of the test.
//       CHECK: #[[MAP:.+]] =  affine_map<(d0, d1) -> (d0 + d1 * 4)>
//       CHECK: func @indexed_consumer_reshape_producer_fusion
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: i32, %[[ARG4:[a-zA-Z0-9_]+]]: i32,
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: i32)
//   CHECK-DAG:     %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:     %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:     %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:     %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:     %[[T3:.+]] = affine.apply #[[MAP]](%[[IDX1]], %[[IDX0]])
//       CHECK:     %[[T4:.+]] = arith.muli %[[ARG3]], %[[ARG4]]
//       CHECK:     %[[T5:.+]] = arith.index_cast %[[T3]]
//       CHECK:     %[[T6:.+]] = arith.addi %[[T4]], %[[T5]]
//       CHECK:     %[[T7:.+]] = arith.index_cast %[[IDX2]]
//       CHECK:     %[[T8:.+]] = arith.addi %[[T6]], %[[T7]]
//       CHECK:     %[[T9:.+]] = arith.index_cast %[[IDX3]]
//       CHECK:     %[[T10:.+]] = arith.addi %[[T8]], %[[T9]]
//       CHECK:     linalg.yield %[[T10]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @indexed_producer_reshape_consumer_fusion(%arg0 : tensor<?x?xi32>,
                                         %arg1 : tensor<?x?xi32>, 
                                         %sz0: index, %sz1: index) ->
                                         tensor<?x?x4x5xi32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%arg0 : tensor<?x?xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %s: i32):
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %1 = arith.muli %arg3, %arg4 : i32
      %2 = arith.index_cast %idx0 : index to i32
      %3 = arith.addi %1, %2 : i32
      %4 = arith.index_cast %idx1 : index to i32
      %5 = arith.addi %3, %4 : i32
      linalg.yield %5 : i32
  } -> tensor<?x?xi32>
  %1 = tensor.expand_shape %0 [[0], [1, 2, 3]] output_shape [%sz0, %sz1, 4, 5] :
    tensor<?x?xi32> into tensor<?x?x4x5xi32>
  return %1 : tensor<?x?x4x5xi32>
}

// Only check the body in the indexed version of the test.
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 4)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 5)>
//       CHECK: func @indexed_producer_reshape_consumer_fusion
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]]: i32, %[[ARG4:[a-zA-Z0-9_]+]]: i32,
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9_]+]]: i32)
//   CHECK-DAG:     %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:     %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:     %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:     %[[IDX3:.+]] = linalg.index 3 : index
//       CHECK:     %[[T1:.+]] = affine.apply #[[MAP1]](%[[IDX2]], %[[IDX1]])
//       CHECK:     %[[T2:.+]] = affine.apply #[[MAP2]](%[[IDX3]], %[[T1]])
//       CHECK:     %[[T4:.+]] = arith.muli %[[ARG3]], %[[ARG4]]
//       CHECK:     %[[T5:.+]] = arith.index_cast %[[IDX0]]
//       CHECK:     %[[T6:.+]] = arith.addi %[[T4]], %[[T5]]
//       CHECK:     %[[T7:.+]] = arith.index_cast %[[T2]]
//       CHECK:     %[[T8:.+]] = arith.addi %[[T6]], %[[T7]]
//       CHECK:     linalg.yield %[[T8]]

// -----

func.func @reshape_as_consumer_permutation
  (%a : tensor<210x6x4xi32>, %b : tensor<210x4xi32>)
    -> tensor<2x3x4x5x6x7xi32> {
  %shape = tensor.empty() : tensor<6x4x210xi32>
  %c = linalg.generic {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
         iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%a, %b : tensor<210x6x4xi32>, tensor<210x4xi32>)
          outs(%shape : tensor<6x4x210xi32>) {
       ^bb0(%arg3 : i32, %arg4: i32, %s: i32):
         %idx0 = linalg.index 0 : index
         %idx1 = linalg.index 1 : index
         %idx2 = linalg.index 2 : index
         %1 = arith.addi %arg3, %arg4 : i32
         %2 = arith.index_cast %idx0 : index to i32
         %3 = arith.addi %1, %2 : i32
         %4 = arith.index_cast %idx1 : index to i32
         %5 = arith.addi %3, %4 : i32
         %6 = arith.index_cast %idx2 : index to i32
         %7 = arith.addi %5, %6 : i32
         linalg.yield %7 : i32
       } -> tensor<6x4x210xi32>
  %d = tensor.expand_shape %c [[0, 1], [2], [3, 4, 5]] output_shape [2, 3, 4, 5, 6, 7] : tensor<6x4x210xi32> into tensor<2x3x4x5x6x7xi32>
  return %d : tensor<2x3x4x5x6x7xi32>
}

// -----

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//   CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 3)>
//   CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 6)>
//   CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 7)>
//       CHECK: func @reshape_as_consumer_permutation
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<210x6x4xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<210x4xi32>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty()
//       CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3, 4], [5]] output_shape [5, 6, 7, 2, 3, 4] : tensor<210x6x4xi32> into tensor<5x6x7x2x3x4xi32>
//       CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3]] output_shape [5, 6, 7, 4] : tensor<210x4xi32> into tensor<5x6x7x4xi32>
//       CHECK:   %[[T3:.+]] = tensor.expand_shape %[[VAL_0]] {{\[\[}}0, 1], [2], [3, 4, 5]] output_shape [2, 3, 4, 5, 6, 7] : tensor<6x4x210xi32> into tensor<2x3x4x5x6x7xi32>
//       CHECK:   %[[T4:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
//  CHECK-SAME:     ins(%[[T1]], %[[T2]] : tensor<5x6x7x2x3x4xi32>, tensor<5x6x7x4xi32>)
//  CHECK-SAME:     outs(%[[T3]] : tensor<2x3x4x5x6x7xi32>)
//       CHECK:   ^{{.+}}(
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9_]+]]: i32, %[[ARG9:[a-zA-Z0-9_]+]]: i32,
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9_]+]]: i32)
//   CHECK-DAG:       %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:       %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:       %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:       %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:       %[[IDX4:.+]] = linalg.index 4 : index
//   CHECK-DAG:       %[[IDX5:.+]] = linalg.index 5 : index
//   CHECK-DAG:       %[[T5:.+]] = affine.apply #[[MAP3]](%[[IDX1]], %[[IDX0]])
//   CHECK-DAG:       %[[T6:.+]] = affine.apply #[[MAP4]](%[[IDX3]], %[[IDX2]])
//   CHECK-DAG:       %[[T7:.+]] = affine.apply #[[MAP5]](%[[IDX4]], %[[T6]])
//   CHECK-DAG:       %[[T8:.+]] = arith.addi %[[ARG8]], %[[ARG9]]
//       CHECK:       %[[T9:.+]] = arith.index_cast %[[T5]]
//       CHECK:       %[[T10:.+]] = arith.addi %[[T8]], %[[T9]]
//       CHECK:       %[[T11:.+]] = arith.index_cast %[[T7]]
//       CHECK:       %[[T12:.+]] = arith.addi %[[T10]], %[[T11]]
//       CHECK:       %[[T13:.+]] = arith.index_cast %[[IDX5]]
//       CHECK:       %[[T14:.+]] = arith.addi %[[T12]], %[[T13]]

// -----

func.func @reshape_as_producer_projected_permutation(
    %arg0 : tensor<33x8x?xi32>, %shape : tensor<264x?x4xi32>) -> tensor<264x?x4xi32>
{
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]]
    : tensor<33x8x?xi32> into tensor<264x?xi32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%0 : tensor<264x?xi32>)
    outs(%shape : tensor<264x?x4xi32>) {
  ^bb0(%arg1: i32, %s: i32):
    %idx0 = linalg.index 0 : index
    %idx1 = linalg.index 1 : index
    %idx2 = linalg.index 2 : index
    %2 = arith.index_cast %idx0 : index to i32
    %3 = arith.addi %arg1, %2 : i32
    %4 = arith.index_cast %idx1 : index to i32
    %5 = arith.addi %3, %4 : i32
    %6 = arith.index_cast %idx2 : index to i32
    %7 = arith.addi %5, %6 : i32
    linalg.yield %7 : i32
  } -> tensor<264x?x4xi32>
  return %1 : tensor<264x?x4xi32>
}

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 8)>
//       CHECK: @reshape_as_producer_projected_permutation
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<33x8x?xi32>
//       CHECK:   %[[RES:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:     ins(%[[ARG0]] : tensor<33x8x?xi32>)
//       CHECK:   ^{{.+}}(
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:       %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:       %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:       %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:       %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:       %[[T0:.+]] = affine.apply #[[MAP2]](%[[IDX1]], %[[IDX0]])
//       CHECK:       %[[T1:.+]] = arith.index_cast %[[T0]] : index to i32
//       CHECK:       %[[T2:.+]] = arith.addi %[[ARG1]], %[[T1]] : i32
//       CHECK:       %[[T3:.+]] = arith.index_cast %[[IDX2]] : index to i32
//       CHECK:       %[[T4:.+]] = arith.addi %[[T2]], %[[T3]] : i32
//       CHECK:       %[[T5:.+]] = arith.index_cast %[[IDX3]] : index to i32
//       CHECK:       %[[T6:.+]] = arith.addi %[[T4]], %[[T5]] : i32
//       CHECK:       linalg.yield %[[T6]] : i32
//       CHECK:    %[[RES2:.+]] = tensor.collapse_shape %[[RES]]
//  CHECK-SAME:      [0, 1], [2], [3]
//  CHECK-SAME:    : tensor<33x8x?x4xi32> into tensor<264x?x4xi32>
//       CHECK:  return %[[RES2]] : tensor<264x?x4xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
func.func @generic_op_reshape_consumer_fusion_projected(%arg0 : tensor<?x?xf32>,
                                                   %arg1 : tensor<?x?xf32>,
                                                   %sz0: index, %sz1: index) ->
                                                   tensor<?x?x4x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map1],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2, 3]] output_shape [%sz0, %sz1, 4, 5] :
    tensor<?x?xf32> into tensor<?x?x4x5xf32>
  return %1 : tensor<?x?x4x5xf32>
}

//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
//      CHECK: func @generic_op_reshape_consumer_fusion_projected
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index
//      CHECK:   %[[C20:.+]] = arith.constant 20 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C20]] : index
//      CHECK:   %[[T0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3]] output_shape [%[[VAL_0]], 4, 5, %[[DIM_0]]] : tensor<?x?xf32> into tensor<?x4x5x?xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_1]], %[[C20]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3]] output_shape [%[[VAL_1]], 4, 5, %[[DIM_2]]] : tensor<?x?xf32> into tensor<?x4x5x?xf32>
//      CHECK:   %[[DIM_4:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_5:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_5]], %[[C20]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_4]], %[[VAL_2]], 4, 5] : tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP5]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<?x4x5x?xf32>, tensor<?x4x5x?xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x?x4x5xf32>)
//      CHECK:   return %[[T3]] : tensor<?x?x4x5xf32>

// -----

func.func @no_fuse_dynamic_dims(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?xf32>
  %2 = tensor.empty(%1) : tensor<?xf32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%0 : tensor<?xf32>) outs(%2 : tensor<?xf32>) {
      ^bb0(%arg1 : f32, %arg2: f32):
        %4 = arith.addf %arg1, %arg1 : f32
        linalg.yield %4 : f32
    } -> tensor<?xf32>
  return %3 : tensor<?xf32>
}

//      CHECK: func @no_fuse_dynamic_dims
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//      CHECK:   %[[RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]] : tensor<?xf32>)
//      CHECK:   return %[[GENERIC]]

// -----

func.func @no_fuse_mismatched_dynamism(%arg0: tensor<2x1xi64>, %arg1: tensor<?xi64>) -> tensor<2xi64> {
  %0 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x1xi64> into tensor<2xi64>
  %1 = tensor.empty() : tensor<2xi64>
  %2 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%0, %arg1 : tensor<2xi64>, tensor<?xi64>)
    outs(%1 : tensor<2xi64>) {
  ^bb0(%arg4: i64, %arg5: i64, %arg6: i64):
    %3 = arith.addi %arg4, %arg5 : i64
    linalg.yield %3 : i64
  } -> tensor<2xi64>
  return %2 : tensor<2xi64>
}

//      CHECK: func @no_fuse_mismatched_dynamism
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x1xi64>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<?xi64>
//      CHECK:   %[[RESHAPE:.+]] = tensor.collapse_shape %[[ARG0]]
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       ins(%[[RESHAPE]], %[[ARG1]] : tensor<2xi64>, tensor<?xi64>)
//      CHECK:   return %[[GENERIC]]

// -----

func.func @reshape_as_consumer_permutation_with_multiple_results
  (%a : tensor<?x?x?xf32>, %b : tensor<?x?xf32>, %sz0: index, 
   %sz1: index, %sz2: index, %sz3: index, %sz4: index)
    -> (tensor<?x2x?x3x4x?xf32>, tensor<?x?x2x3x4x?xf32>) {
  %c:2 = linalg.generic {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
                          affine_map<(d0, d1, d2) -> (d2, d0, d1)>],
         iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?xf32>)
         outs(%a, %a : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
       ^bb0(%arg0 : f32, %arg1: f32, %s: f32, %t : f32):
         %1 = arith.addf %arg0, %arg1 : f32
         linalg.yield %1, %1 : f32, f32
       } -> (tensor<?x?x?xf32>, tensor<?x?x?xf32>)
  %d = tensor.expand_shape %c#0 [[0, 1], [2], [3, 4, 5]] output_shape [%sz0, 2, %sz1, 3, 4, %sz2] : tensor<?x?x?xf32> into tensor<?x2x?x3x4x?xf32>
  %e = tensor.expand_shape %c#1 [[0], [1, 2], [3, 4, 5]] output_shape [%sz3, %sz4, 2, 3, 4, %sz2] : tensor<?x?x?xf32> into tensor<?x?x2x3x4x?xf32>
  return %d, %e : tensor<?x2x?x3x4x?xf32>, tensor<?x?x2x3x4x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d1, d2, d3, d4)>
//      CHECK: func @reshape_as_consumer_permutation_with_multiple_results
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index, %[[SZ2:.+]]: index, %[[SZ3:.+]]: index, %[[SZ4:.+]]: index
//       CHECK:   %[[C12:.+]] = arith.constant 12 : index
//       CHECK:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
//       CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C12]] : index
//       CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_0]], %[[C2]] : index
//       CHECK:   %[[RESHAPE0:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1, 2], [3, 4], [5]] output_shape [3, 4, %[[VAL_0]], %[[VAL_1]], 2, %[[DIM_1]]] : tensor<?x?x?xf32> into tensor<3x4x?x?x2x?xf32>
//       CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//       CHECK:   %[[DIM_3:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//       CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_2]], %[[C12]] : index
//       CHECK:   %[[RESHAPE1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3]] output_shape [3, 4, %[[VAL_2]], %[[DIM_3]]] : tensor<?x?xf32> into tensor<3x4x?x?xf32>
//       CHECK:   %[[DIM_5:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_6:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_7:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
//       CHECK:   %[[VAL_3:.+]] = arith.divsi %[[DIM_5]], %[[C2]] : index
//       CHECK:   %[[VAL_4:.+]] = arith.divsi %[[DIM_7]], %[[C12]] : index
//       CHECK:   %[[RESHAPE2:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0, 1], [2], [3, 4, 5]] output_shape [%[[VAL_3]], 2, %[[DIM_6]], 3, 4, %[[VAL_4]]] : tensor<?x?x?xf32> into tensor<?x2x?x3x4x?xf32>
//       CHECK:   %[[DIM_9:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_10:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
//       CHECK:   %[[DIM_11:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
//       CHECK:   %[[VAL_5:.+]] = arith.divsi %[[DIM_10]], %[[C2]] : index
//       CHECK:   %[[VAL_6:.+]] = arith.divsi %[[DIM_11]], %[[C12]] : index
//       CHECK:   %[[RESHAPE3:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2], [3, 4, 5]] output_shape [%[[DIM_9]], %[[VAL_5]], 2, 3, 4, %[[VAL_6]]] : tensor<?x?x?xf32> into tensor<?x?x2x3x4x?xf32>
//       CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
//  CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
//  CHECK-SAME:      ins(%[[RESHAPE0]], %[[RESHAPE1]] :
//  CHECK-SAME:      outs(%[[RESHAPE2]], %[[RESHAPE3]] :
//       CHECK:  return %[[GENERIC]]#0, %[[GENERIC]]#1

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @multi_result_op_expansion(%arg0: tensor<512xf32>, %arg1: tensor<512xf32>,
      %arg2: tensor<512xf32>, %arg3: tensor<200x512xf32>) -> tensor<25x8x1x512xf32> {
    %0:2 = linalg.generic {
        indexing_maps = [#map0, #map0, #map0, #map1],
        iterator_types = ["parallel", "parallel"]}
        ins(%arg0, %arg1 : tensor<512xf32>, tensor<512xf32>)
        outs(%arg2, %arg3 : tensor<512xf32>, tensor<200x512xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32):
        %2 = arith.addf %arg4, %arg5 : f32
        linalg.yield %2, %2 : f32, f32
      } -> (tensor<512xf32>, tensor<200x512xf32>)
    %1 = tensor.expand_shape %0#1 [[0, 1, 2], [3]] output_shape [25, 8, 1, 512] : tensor<200x512xf32> into tensor<25x8x1x512xf32>
    return %1 : tensor<25x8x1x512xf32>
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func.func @multi_result_op_expansion(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<512xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<512xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<512xf32>
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: tensor<200x512xf32>
//      CHECK:     %[[OUTS:.+]] = tensor.expand_shape %[[ARG3]] {{\[\[}}0, 1, 2], [3]] output_shape [25, 8, 1, 512] : tensor<200x512xf32> into tensor<25x8x1x512xf32>
//      CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]], #[[MAP1]]]
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]], %[[OUTS]] :
//      CHECK:   return %[[GENERIC]]#1

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @generic_op_reshape_consumer_fusion_reduction(%arg0 : tensor<?x?xf32>,
                                                        %arg1 : tensor<?x?xf32>,
                                                        %arg2 : tensor<?x?xf32>,
                                                        %sz0: index,
                                                        %sz1: index) ->
                                                        tensor<?x?x4x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map1, #map2],
     iterator_types = ["parallel", "parallel", "reduction"]}
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2, 3]] output_shape [%sz0, %sz1, 4, 5] :
    tensor<?x?xf32> into tensor<?x?x4x5xf32>
  return %1 : tensor<?x?x4x5xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
//      CHECK: func @generic_op_reshape_consumer_fusion_reduction
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index
//      CHECK:   %[[C20:.+]] = arith.constant 20 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C20]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2], [3]] output_shape [%[[VAL_0]], 4, 5, %[[DIM_0]]] : tensor<?x?xf32> into tensor<?x4x5x?xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG2]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_2]], %[[C20]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_1]], %[[VAL_1]], 4, 5] : tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T1]] : tensor<?x?xf32>, tensor<?x4x5x?xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x?x4x5xf32>)
//      CHECK:   return %[[T3]] : tensor<?x?x4x5xf32>

// -----

#map0 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @generic_op_reshape_producer_fusion_with_reduction(%arg0 : tensor<?x7x?x8xf32>,
                                         %arg1 : tensor<?x4x?xf32>,
                                         %arg2 : tensor<?x?xf32>) ->
                                         tensor<?x?xf32>
{
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] :
    tensor<?x7x?x8xf32> into tensor<?x?xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map1, #map2],
     iterator_types = ["parallel", "reduction", "parallel"]}
       ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x4x?xf32>)
       outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

//  CHECK-DAG: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d0, d1)>
//  CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//  CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
//      CHECK: func @generic_op_reshape_producer_fusion_with_reduction
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x7x?x8xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x4x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C7:.+]] = arith.constant 7 : index
//      CHECK:   %[[C8:.+]] = arith.constant 8 : index
//      CHECK:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x4x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<?x4x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C8]] : index
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_0]], %[[C7]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1], [2], [3, 4]] output_shape [%[[VAL_0]], 8, 4, %[[VAL_1]], 7] : tensor<?x4x?xf32> into tensor<?x8x4x?x7xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG2]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_1]], %[[C8]] : index
//      CHECK:   %[[VAL_3:.+]] = arith.divsi %[[DIM_2]], %[[C7]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1], [2, 3]] output_shape [%[[VAL_2]], 8, %[[VAL_3]], 7] : tensor<?x?xf32> into tensor<?x8x?x7xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "reduction", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T1]] : tensor<?x7x?x8xf32>, tensor<?x8x4x?x7xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x8x?x7xf32>)
//      CHECK:   %[[T4:.+]] = tensor.collapse_shape %[[T3]]
// CHECK-SAME:     [0, 1], [2, 3]
// CHECK-SAME:     tensor<?x8x?x7xf32> into tensor<?x?xf32>
//      CHECK:   return %[[T4]]

// -----

func.func @linalg_add_reshape_consumer_fusion(%arg0 : tensor<?x?xf32>,
                                              %arg1 : tensor<?x?xf32>,
                                              %arg2 : tensor<?x?xf32>,
                                              %sz0: index,
                                              %sz1: index) ->
                                              tensor<?x?x4x5xf32>
{
  %0 = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2, 3]] output_shape [%sz0, %sz1, 4, 5] :
    tensor<?x?xf32> into tensor<?x?x4x5xf32>
  return %1 : tensor<?x?x4x5xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @linalg_add_reshape_consumer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[SZ0:.+]]: index, %[[SZ1:.+]]: index
//      CHECK:   %[[C20:.+]] = arith.constant 20 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM_0]], %[[C20]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM]], %[[VAL_0]], 4, 5] : tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_2]], %[[C20]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_1]], %[[VAL_1]], 4, 5] : tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[DIM_4:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_5:.+]] = tensor.dim %[[ARG2]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_5]], %[[C20]] : index
//      CHECK:   %[[T3:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0], [1, 2, 3]] output_shape [%[[DIM_4]], %[[VAL_2]], 4, 5] : tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[T4:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T1]], %[[T2]] : tensor<?x?x4x5xf32>, tensor<?x?x4x5xf32>)
// CHECK-SAME:     outs(%[[T3]] : tensor<?x?x4x5xf32>)
//      CHECK:   return %[[T4]] : tensor<?x?x4x5xf32>

// -----

#map0 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @linalg_add_reshape_producer_fusion(%arg0 : tensor<?x7x?x8xf32>,
                                              %arg1 : tensor<?x?xf32>,
                                              %arg2 : tensor<?x?xf32>) ->
                                              tensor<?x?xf32>
{
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] :
    tensor<?x7x?x8xf32> into tensor<?x?xf32>
  %1 = linalg.add ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

//  CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @linalg_add_reshape_producer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x7x?x8xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[C8:.+]] = arith.constant 8 : index
//      CHECK:   %[[C7:.+]] = arith.constant 7 : index
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[DIM:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_0:.+]] = tensor.dim %[[ARG1]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_0:.+]] = arith.divsi %[[DIM]], %[[C7]] : index
//      CHECK:   %[[VAL_1:.+]] = arith.divsi %[[DIM_0]], %[[C8]] : index
//      CHECK:   %[[T1:.+]] = tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1], [2, 3]] output_shape [%[[VAL_0]], 7, %[[VAL_1]], 8] : tensor<?x?xf32> into tensor<?x7x?x8xf32>
//      CHECK:   %[[DIM_1:.+]] = tensor.dim %[[ARG2]], %[[C0]] : tensor<?x?xf32>
//      CHECK:   %[[DIM_2:.+]] = tensor.dim %[[ARG2]], %[[C1]] : tensor<?x?xf32>
//      CHECK:   %[[VAL_2:.+]] = arith.divsi %[[DIM_1]], %[[C7]] : index
//      CHECK:   %[[VAL_3:.+]] = arith.divsi %[[DIM_2]], %[[C8]] : index
//      CHECK:   %[[T2:.+]] = tensor.expand_shape %[[ARG2]] {{\[\[}}0, 1], [2, 3]] output_shape [%[[VAL_2]], 7, %[[VAL_3]], 8] : tensor<?x?xf32> into tensor<?x7x?x8xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T1]] : tensor<?x7x?x8xf32>, tensor<?x7x?x8xf32>)
// CHECK-SAME:     outs(%[[T2]] : tensor<?x7x?x8xf32>)
//      CHECK:   %[[T4:.+]] = tensor.collapse_shape %[[T3]]
// CHECK-SAME:     [0, 1], [2, 3]
// CHECK-SAME:     tensor<?x7x?x8xf32> into tensor<?x?xf32>
//      CHECK:   return %[[T4]]

// -----

func.func @fuse_by_expanding_pad(%arg0 : tensor<2x3x4x5x6x7x8x9xi32>) -> tensor<8x12x17x336x14xi32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] : tensor<2x3x4x5x6x7x8x9xi32> into tensor<2x12x5x336x9xi32>
  %cst = arith.constant 0 : i32
  %padded_0 = tensor.pad %collapse low[1, 0, 8, 0, 3] high[5, 0, 4, 0, 2] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : i32
  } : tensor<2x12x5x336x9xi32> to tensor<8x12x17x336x14xi32>
  return %padded_0 : tensor<8x12x17x336x14xi32>
}
//      CHECK: func @fuse_by_expanding_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x4x5x6x7x8x9xi32>)
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]]
// CHECK-SAME:       low[1, 0, 0, 8, 0, 0, 0, 3] high[5, 0, 0, 4, 0, 0, 0, 2]
//      CHECK:       tensor<2x3x4x5x6x7x8x9xi32> to tensor<8x3x4x17x6x7x8x14xi32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[PAD]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]]
// CHECK-SAME:       : tensor<8x3x4x17x6x7x8x14xi32> into tensor<8x12x17x336x14xi32>
//      CHECK:   return %[[COLLAPSE]]

// -----

func.func @no_fuse_by_expanding_pad(%arg0 : tensor<2x3x4x5x6x7x8x9xi32>) -> tensor<8x12x17x339x14xi32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4, 5, 6], [7]] : tensor<2x3x4x5x6x7x8x9xi32> into tensor<2x12x5x336x9xi32>
  %cst = arith.constant 0 : i32
  %padded_0 = tensor.pad %collapse low[1, 0, 8, 0, 3] high[5, 0, 4, 3, 2] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
    tensor.yield %cst : i32
  } : tensor<2x12x5x336x9xi32> to tensor<8x12x17x339x14xi32>
  return %padded_0 : tensor<8x12x17x339x14xi32>
}
//      CHECK: func @no_fuse_by_expanding_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x3x4x5x6x7x8x9xi32>)
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3], [4, 5, 6], [7]]
// CHECK-SAME:       : tensor<2x3x4x5x6x7x8x9xi32> into tensor<2x12x5x336x9xi32>
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[COLLAPSE]]
// CHECK-SAME:       low[1, 0, 8, 0, 3] high[5, 0, 4, 3, 2]
//      CHECK:       tensor<2x12x5x336x9xi32> to tensor<8x12x17x339x14xi32>
//      CHECK:   return %[[PAD]]

// -----

func.func @fuse_by_expanding_dynamic_pad(%arg0 : tensor<?x?x?x?x?x?xi32>, %l0: index, %l1: index, %h0: index, %h1: index) -> tensor<?x?x?x?xi32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3], [4, 5]] : tensor<?x?x?x?x?x?xi32> into tensor<?x?x?x?xi32>
  %cst = arith.constant 0 : i32
  %padded_0 = tensor.pad %collapse low[%l0, 0, %l1, 0] high[%h0, 0, %h1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %cst : i32
  } : tensor<?x?x?x?xi32> to tensor<?x?x?x?xi32>
  return %padded_0 : tensor<?x?x?x?xi32>
}
//      CHECK: func @fuse_by_expanding_dynamic_pad(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?x?x?xi32>
// CHECK-SAME:   %[[L0:.+]]: index, %[[L1:.+]]: index, %[[H0:.+]]: index, %[[H1:.+]]: index
//      CHECK:   %[[PAD:.+]] = tensor.pad %[[ARG0]]
// CHECK-SAME:       low[%[[L0]], 0, 0, %[[L1]], 0, 0] high[%[[H0]], 0, 0, %[[H1]], 0, 0]
//      CHECK:       tensor<?x?x?x?x?x?xi32> to tensor<?x?x?x?x?x?xi32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[PAD]] {{\[}}[0], [1, 2], [3], [4, 5]]
// CHECK-SAME:       : tensor<?x?x?x?x?x?xi32> into tensor<?x?x?x?xi32>
//      CHECK:   return %[[COLLAPSE]]

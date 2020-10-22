// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?x?xf32>,
                                         %arg1 : tensor<?x?x?xf32>) ->
                                         tensor<?x?x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(i, j, k, l) -> (i)>,
                                    affine_map<(i, j, k, l) -> (j, k)>,
                                    affine_map<(i, j, k, l) -> (l)>] :
    tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map1, #map1],
     iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
//      CHECK: func @generic_op_reshape_producer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_reshape %[[ARG1]]
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
//      CHECK:   %[[T1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP4]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T0]] : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>)
//      CHECK:   %[[T2:.+]] = linalg.tensor_reshape
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
//      CHECK:   return %[[T2]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?xf32>,
                                         %arg1 : tensor<?x?xf32>) ->
                                         tensor<?x?x4x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?xf32> into tensor<?x?x4x5xf32>
  return %1 : tensor<?x?x4x5xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @generic_op_reshape_consumer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[T1:.+]] = linalg.tensor_reshape %[[ARG1]]
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x?x4x5xf32>
//      CHECK:   %[[T2:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<?x?x4x5xf32>, tensor<?x?x4x5xf32>)
//      CHECK:   return %[[T2]] : tensor<?x?x4x5xf32>


// -----

func @reshape_as_consumer_permutation
  (%a : tensor<?x?x?xf32>, %b : tensor<?x?xf32>)
    -> tensor<?x?x?x?x?x?xf32> {
  %c = linalg.generic {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
         iterator_types = ["parallel", "parallel", "parallel"]}
         ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?xf32>) {
       ^bb0(%arg0 : f32, %arg1: f32):
         %1 = addf %arg0, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?x?xf32>
  %d = linalg.tensor_reshape %c
         [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d2)>,
          affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>]
       : tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
  return %d : tensor<?x?x?x?x?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//  CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//      CHECK: func @reshape_as_consumer_permutation
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     tensor<?x?x?xf32> into tensor<?x?x?x?x?x?xf32>
//      CHECK:   %[[T1:.+]] = linalg.tensor_reshape %[[ARG1]]
// CHECK-SAME:     [#[[MAP3]], #[[MAP4]]]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x?x?x?xf32>
//      CHECK:   %[[T2:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP5]], #[[MAP6]], #[[MAP7]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<?x?x?x?x?x?xf32>, tensor<?x?x?x?xf32>)
//      CHECK:   return %[[T2]] : tensor<?x?x?x?x?x?xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

func @generic_op_reshape_consumer_static(%arg0: tensor<264x4xf32>)
                                            -> tensor<8x33x4xf32> {
  %cst = constant dense<2.000000e+00> : tensor<264x4xf32>
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %cst : tensor<264x4xf32>, tensor<264x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %2 = mulf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<264x4xf32>
  %1 = linalg.tensor_reshape %0 [#map1, #map2] :
    tensor<264x4xf32> into tensor<8x33x4xf32>
  return %1 : tensor<8x33x4xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @generic_op_reshape_consumer_static
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<264x4xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:     [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     tensor<264x4xf32> into tensor<8x33x4xf32>
//      CHECK:   %[[T1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]] : tensor<8x33x4xf32>)
//      CHECK:   return %[[T1]] : tensor<8x33x4xf32>

// -----

func @scalar_reshape(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1xf32>)
                     -> tensor<1x10xf32> {
  %0 = linalg.tensor_reshape %arg1 [] : tensor<1xf32> into tensor<f32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]} ins(%0 : tensor<f32>) {
  ^bb0(%arg2: f32):  // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<10xf32>
  %2 = linalg.tensor_reshape %1 [affine_map<(d0, d1) -> (d0, d1)>]
    : tensor<10xf32> into tensor<1x10xf32>
  return %2 : tensor<1x10xf32>
}

//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> ()>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//      CHECK: func @scalar_reshape
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<1xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_reshape %[[ARG1]] []
// CHECK-SAME:     tensor<1xf32> into tensor<f32>
//      CHECK:   %[[T1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]] : tensor<f32>)
//      CHECK:   return %[[T1]] : tensor<1x10xf32>

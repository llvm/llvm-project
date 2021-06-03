// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops="allow-folding-unit-dim-reshapes=false" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -linalg-fusion-for-tensor-ops="allow-folding-unit-dim-reshapes=true" -split-input-file | FileCheck %s --check-prefix=FOLDUNITDIM
#map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x4x?xf32>,
                                         %arg1 : tensor<?x?x?xf32>) ->
                                         tensor<?x?x?xf32>
{
  %0 = linalg.tensor_collapse_shape %arg0 [[0], [1, 2], [3]] :
    tensor<?x?x4x?xf32> into tensor<?x?x?xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map1, #map1],
     iterator_types = ["parallel", "parallel", "parallel"]}
       ins(%0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
       outs(%0 : tensor<?x?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
//  CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>
//      CHECK: func @generic_op_reshape_producer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x4x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_collapse_shape %[[ARG0]]
// CHECK-SAME:     [0], [1, 2], [3]
//      CHECK:   %[[T1:.+]] = linalg.tensor_expand_shape %[[ARG1]]
// CHECK-SAME:     [0], [1], [2, 3]
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP5]], #[[MAP6]], #[[MAP6]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[ARG0]], %[[T1]] : tensor<?x?x4x?xf32>, tensor<?x?x?x4xf32>)
// CHECK-SAME:     outs(%{{.+}} : tensor<?x?x?x4xf32>)
//      CHECK:   %[[T4:.+]] = linalg.tensor_collapse_shape %[[T3]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     tensor<?x?x?x4xf32> into tensor<?x?x?xf32>
//      CHECK:   return %[[T4]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?xf32>,
                                         %arg1 : tensor<?x?xf32>) ->
                                         tensor<?x4x?x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = linalg.tensor_expand_shape %0 [[0], [1, 2, 3]] :
    tensor<?x?xf32> into tensor<?x4x?x5xf32>
  return %1 : tensor<?x4x?x5xf32>
}

//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @generic_op_reshape_consumer_fusion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>)
//      CHECK:   %[[T0:.+]] = linalg.tensor_expand_shape %[[ARG0]]
// CHECK-SAME:     [0], [1, 2, 3]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x4x?x5xf32>
//      CHECK:   %[[T1:.+]] = linalg.tensor_expand_shape %[[ARG1]]
// CHECK-SAME:     [0], [1, 2, 3]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x4x?x5xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<?x4x?x5xf32>, tensor<?x4x?x5xf32>)
// CHECK-SAME:     outs(%{{.+}} : tensor<?x4x?x5xf32>)
//      CHECK:   return %[[T3]] : tensor<?x4x?x5xf32>


// -----

func @reshape_as_consumer_permutation
  (%a : tensor<?x?x?xf32>, %b : tensor<?x?xf32>)
    -> tensor<?x2x?x3x4x?xf32> {
  %c = linalg.generic {
         indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>,
                          affine_map<(d0, d1, d2) -> (d1, d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d2, d1)>],
         iterator_types = ["parallel", "parallel", "parallel"]}
          ins(%a, %b : tensor<?x?x?xf32>, tensor<?x?xf32>)
         outs(%a : tensor<?x?x?xf32>) {
       ^bb0(%arg0 : f32, %arg1: f32, %s: f32):
         %1 = addf %arg0, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?x?xf32>
  %d = linalg.tensor_expand_shape %c [[0, 1], [2], [3, 4, 5]]
       : tensor<?x?x?xf32> into tensor<?x2x?x3x4x?xf32>
  return %d : tensor<?x2x?x3x4x?xf32>
}
//  CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//  CHECK-DAG: #[[MAP9:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//  CHECK-DAG: #[[MAP10:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//      CHECK: func @reshape_as_consumer_permutation
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_expand_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3, 4], [5]
// CHECK-SAME:     tensor<?x?x?xf32> into tensor<3x4x?x?x2x?xf32>
//      CHECK:   %[[T1:.+]] = linalg.tensor_expand_shape %[[ARG1]]
// CHECK-SAME:     [0, 1, 2], [3]
// CHECK-SAME:     tensor<?x?xf32> into tensor<3x4x?x?xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP8]], #[[MAP9]], #[[MAP10]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<3x4x?x?x2x?xf32>, tensor<3x4x?x?xf32>)
// CHECK-SAME:     outs(%{{.+}} : tensor<?x2x?x3x4x?xf32>)
//      CHECK:   return %[[T3]] : tensor<?x2x?x3x4x?xf32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

func @generic_op_reshape_consumer_static(%arg0: tensor<264x4xf32>)
                                            -> tensor<8x33x4xf32> {
  %cst = constant dense<2.000000e+00> : tensor<264x4xf32>
  %0 = linalg.init_tensor [264, 4] : tensor<264x4xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %cst : tensor<264x4xf32>, tensor<264x4xf32>)
       outs(%0 : tensor<264x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %s: f32):  // no predecessors
      %2 = mulf %arg1, %arg2 : f32
      linalg.yield %2 : f32
    } -> tensor<264x4xf32>
  %2 = linalg.tensor_expand_shape %1 [[0, 1], [2]] :
    tensor<264x4xf32> into tensor<8x33x4xf32>
  return %2 : tensor<8x33x4xf32>
}

//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @generic_op_reshape_consumer_static
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<264x4xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_expand_shape %[[ARG0]]
// CHECK-SAME:     [0, 1], [2]
// CHECK-SAME:     tensor<264x4xf32> into tensor<8x33x4xf32>
//      CHECK:   %[[T1:.+]] = linalg.init_tensor [8, 33, 4]
//      CHECK:   %[[T2:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP2]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]] : tensor<8x33x4xf32>)
// CHECK-SAME:     outs(%[[T1]] : tensor<8x33x4xf32>)
//      CHECK:   return %[[T2]] : tensor<8x33x4xf32>

// -----

#map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
func @indexed_consumer_reshape_producer_fusion(%arg0 : tensor<?x?x4x?xi32>,
                                         %arg1 : tensor<?x?x?xi32>) ->
                                         tensor<?x?x?xi32>
{
  %0 = linalg.tensor_collapse_shape %arg0 [[0], [1, 2], [3]]:
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
      %1 = muli %arg3, %arg4 : i32
      %2 = index_cast %idx0 : index to i32
      %3 = addi %1, %2 : i32
      %4 = index_cast %idx1 : index to i32
      %5 = addi %3, %4 : i32
      %6 = index_cast %idx2 : index to i32
      %7 = addi %5, %6 : i32
      linalg.yield %7 : i32
  } -> tensor<?x?x?xi32>
  return %1 : tensor<?x?x?xi32>
}

// Only check the body in the indexed version of the test.
//       CHECK: #[[MAP:.+]] =  affine_map<(d0, d1) -> (d0 + d1 * 4)>
//       CHECK: func @indexed_consumer_reshape_producer_fusion
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: i32, %[[ARG4:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:     %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:     %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:     %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:     %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:     %[[T3:.+]] = affine.apply #[[MAP]](%[[IDX1]], %[[IDX0]])
//       CHECK:     %[[T4:.+]] = muli %[[ARG3]], %[[ARG4]]
//       CHECK:     %[[T5:.+]] = index_cast %[[T3]]
//       CHECK:     %[[T6:.+]] = addi %[[T4]], %[[T5]]
//       CHECK:     %[[T7:.+]] = index_cast %[[IDX2]]
//       CHECK:     %[[T8:.+]] = addi %[[T6]], %[[T7]]
//       CHECK:     %[[T9:.+]] = index_cast %[[IDX3]]
//       CHECK:     %[[T10:.+]] = addi %[[T8]], %[[T9]]
//       CHECK:     linalg.yield %[[T10]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @indexed_producer_reshape_consumer_fusion(%arg0 : tensor<?x?xi32>,
                                         %arg1 : tensor<?x?xi32>) ->
                                         tensor<?x?x4x5xi32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%arg0 : tensor<?x?xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %s: i32):       // no predecessors
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %1 = muli %arg3, %arg4 : i32
      %2 = index_cast %idx0 : index to i32
      %3 = addi %1, %2 : i32
      %4 = index_cast %idx1 : index to i32
      %5 = addi %3, %4 : i32
      linalg.yield %5 : i32
  } -> tensor<?x?xi32>
  %1 = linalg.tensor_expand_shape %0 [[0], [1, 2, 3]] :
    tensor<?x?xi32> into tensor<?x?x4x5xi32>
  return %1 : tensor<?x?x4x5xi32>
}

// Only check the body in the indexed version of the test.
//       CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 5 + d2 * 20)>
//       CHECK: func @indexed_producer_reshape_consumer_fusion
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: i32, %[[ARG4:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:     %[[ARG5:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:     %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:     %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:     %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:     %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:     %[[T3:.+]] = affine.apply #[[MAP]](%[[IDX3]], %[[IDX2]], %[[IDX1]])
//       CHECK:     %[[T4:.+]] = muli %[[ARG3]], %[[ARG4]]
//       CHECK:     %[[T5:.+]] = index_cast %[[IDX0]]
//       CHECK:     %[[T6:.+]] = addi %[[T4]], %[[T5]]
//       CHECK:     %[[T7:.+]] = index_cast %[[T3]]
//       CHECK:     %[[T8:.+]] = addi %[[T6]], %[[T7]]
//       CHECK:     linalg.yield %[[T8]]

// -----

func @reshape_as_consumer_permutation
  (%a : tensor<210x6x4xi32>, %b : tensor<210x4xi32>)
    -> tensor<2x3x4x5x6x7xi32> {
  %shape = linalg.init_tensor [6, 4, 210] : tensor<6x4x210xi32>
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
         %1 = addi %arg3, %arg4 : i32
         %2 = index_cast %idx0 : index to i32
         %3 = addi %1, %2 : i32
         %4 = index_cast %idx1 : index to i32
         %5 = addi %3, %4 : i32
         %6 = index_cast %idx2 : index to i32
         %7 = addi %5, %6 : i32
         linalg.yield %7 : i32
       } -> tensor<6x4x210xi32>
  %d = linalg.tensor_expand_shape %c [[0, 1], [2], [3, 4, 5]]
       : tensor<6x4x210xi32> into tensor<2x3x4x5x6x7xi32>
  return %d : tensor<2x3x4x5x6x7xi32>
}


//   CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d0, d1, d5)>
//   CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, d4, d5)>
//   CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d2, d3, d4)>
//   CHECK-DAG: #[[MAP8:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 3)>
//   CHECK-DAG: #[[MAP9:.+]] = affine_map<(d0, d1, d2) -> (d0 + d1 * 7 + d2 * 42)>
//       CHECK: func @reshape_as_consumer_permutation
//  CHECK-SAME:   %[[ARG0:.+]]: tensor<210x6x4xi32>
//  CHECK-SAME:   %[[ARG1:.+]]: tensor<210x4xi32>
//   CHECK-DAG:   %[[T1:.+]] = linalg.tensor_expand_shape %[[ARG0]]
//  CHECK-SAME:     [0, 1, 2], [3, 4], [5]
//   CHECK-DAG:   %[[T2:.+]] = linalg.tensor_expand_shape %[[ARG1]]
//  CHECK-SAME:     [0, 1, 2], [3]
//   CHECK-DAG:   %[[T0:.+]] = linalg.init_tensor [2, 3, 4, 5, 6, 7]
//       CHECK:   %[[T4:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP5]], #[[MAP6]], #[[MAP7]]]
//  CHECK-SAME:     ins(%[[T1]], %[[T2]] : tensor<5x6x7x2x3x4xi32>, tensor<5x6x7x4xi32>)
//  CHECK-SAME:     outs(%[[T0]] : tensor<2x3x4x5x6x7xi32>)
//       CHECK:   ^{{.+}}(
//  CHECK-SAME:     %[[ARG8:[a-zA-Z0-9]+]]: i32, %[[ARG9:[a-zA-Z0-9]+]]: i32,
//  CHECK-SAME:     %[[ARG10:[a-zA-Z0-9]+]]: i32)
//   CHECK-DAG:       %[[IDX0:.+]] = linalg.index 0 : index
//   CHECK-DAG:       %[[IDX1:.+]] = linalg.index 1 : index
//   CHECK-DAG:       %[[IDX2:.+]] = linalg.index 2 : index
//   CHECK-DAG:       %[[IDX3:.+]] = linalg.index 3 : index
//   CHECK-DAG:       %[[IDX4:.+]] = linalg.index 4 : index
//   CHECK-DAG:       %[[IDX5:.+]] = linalg.index 5 : index
//   CHECK-DAG:       %[[T5:.+]] = affine.apply #[[MAP8]](%[[IDX1]], %[[IDX0]])
//   CHECK-DAG:       %[[T6:.+]] = affine.apply #[[MAP9]](%[[IDX4]], %[[IDX3]], %[[IDX2]])
//   CHECK-DAG:       %[[T7:.+]] = addi %[[ARG8]], %[[ARG9]]
//       CHECK:       %[[T8:.+]] = index_cast %[[T5]]
//       CHECK:       %[[T9:.+]] = addi %[[T7]], %[[T8]]
//       CHECK:       %[[T10:.+]] = index_cast %[[T6]]
//       CHECK:       %[[T11:.+]] = addi %[[T9]], %[[T10]]
//       CHECK:       %[[T12:.+]] = index_cast %[[IDX5]]
//       CHECK:       %[[T13:.+]] = addi %[[T11]], %[[T12]]

// -----

func @reshape_as_producer_projected_permutation(
    %arg0 : tensor<33x8x?xi32>, %shape : tensor<264x?x4xi32>) -> tensor<264x?x4xi32>
{
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]]
    : tensor<33x8x?xi32> into tensor<264x?xi32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%0 : tensor<264x?xi32>)
    outs(%shape : tensor<264x?x4xi32>) {
  ^bb0(%arg1: i32, %s: i32):  // no predecessors
    %idx0 = linalg.index 0 : index
    %idx1 = linalg.index 1 : index
    %idx2 = linalg.index 2 : index
    %2 = index_cast %idx0 : index to i32
    %3 = addi %arg1, %2 : i32
    %4 = index_cast %idx1 : index to i32
    %5 = addi %3, %4 : i32
    %6 = index_cast %idx2 : index to i32
    %7 = addi %5, %6 : i32
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
//       CHECK:       %[[T1:.+]] = index_cast %[[T0]] : index to i32
//       CHECK:       %[[T2:.+]] = addi %[[ARG1]], %[[T1]] : i32
//       CHECK:       %[[T3:.+]] = index_cast %[[IDX2]] : index to i32
//       CHECK:       %[[T4:.+]] = addi %[[T2]], %[[T3]] : i32
//       CHECK:       %[[T5:.+]] = index_cast %[[IDX3]] : index to i32
//       CHECK:       %[[T6:.+]] = addi %[[T4]], %[[T5]] : i32
//       CHECK:       linalg.yield %[[T6]] : i32
//       CHECK:    %[[RES2:.+]] = linalg.tensor_collapse_shape %[[RES]]
//  CHECK-SAME:      [0, 1], [2], [3]
//  CHECK-SAME:    : tensor<33x8x?x4xi32> into tensor<264x?x4xi32>
//       CHECK:  return %[[RES2]] : tensor<264x?x4xi32>

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
func @generic_op_reshape_consumer_fusion_projected(%arg0 : tensor<?x?xf32>,
                                                   %arg1 : tensor<?x?xf32>) ->
                                                   tensor<?x?x4x5xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map1],
     iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg0 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %s: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  %1 = linalg.tensor_expand_shape %0 [[0], [1, 2, 3]] :
    tensor<?x?xf32> into tensor<?x?x4x5xf32>
  return %1 : tensor<?x?x4x5xf32>
}

//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
//      CHECK: func @generic_op_reshape_consumer_fusion_projected
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//      CHECK:   %[[T0:.+]] = linalg.tensor_expand_shape %[[ARG0]]
// CHECK-SAME:     [0, 1, 2], [3]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x4x5x?xf32>
//      CHECK:   %[[T1:.+]] = linalg.tensor_expand_shape %[[ARG1]]
// CHECK-SAME:     [0, 1, 2], [3]
// CHECK-SAME:     tensor<?x?xf32> into tensor<?x4x5x?xf32>
//      CHECK:   %[[T3:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP5]]]
// CHECK-SAME:     ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[T0]], %[[T1]] : tensor<?x4x5x?xf32>, tensor<?x4x5x?xf32>)
// CHECK-SAME:     outs(%{{.+}} : tensor<?x?x4x5xf32>)
//      CHECK:   return %[[T3]] : tensor<?x?x4x5xf32>

// -----

func @unit_dim_reshape_expansion(%arg0 : tensor<1x5xf32>) -> tensor<5x5xf32> {
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1]]
      : tensor<1x5xf32> into tensor<5xf32>
  %1 = linalg.init_tensor [5, 5] : tensor<5x5xf32>
  %2 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%0 : tensor<5xf32>) outs(%1 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}
//      CHECK: func @unit_dim_reshape_expansion
//  CHECK-DAG:   linalg.tensor_collapse_shape
//  CHECK-DAG:   linalg.init_tensor
//      CHECK:   linalg.generic

// -----

func @unit_dim_reshape_collapse(%arg0 : tensor<5xf32>) -> tensor<5x1x5xf32> {
  %0 = linalg.init_tensor [5, 5] : tensor<5x5xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<5xf32>) outs(%0 : tensor<5x5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<5x5xf32>
  %2 = linalg.tensor_expand_shape %1 [[0, 1], [2]]
    : tensor<5x5xf32> into tensor<5x1x5xf32>
  return %2 : tensor<5x1x5xf32>
}
// CHECK: func @unit_dim_reshape_collapse
// CHECK:   linalg.init_tensor
// CHECK:   linalg.generic
// CHECK:   linalg.tensor_expand_shape

// -----

func @unit_dim_reshape_expansion_full
  (%arg0 : tensor<1x?x1x2x1x4xf32>, %arg1 : tensor<?x2x4xf32>)
  -> tensor<?x2x4xf32> {
  %c1 = constant 1 : index
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1, 2], [3, 4], [5]]
    : tensor<1x?x1x2x1x4xf32> into tensor<?x2x4xf32>
  %1 = memref.dim %arg0, %c1 : tensor<1x?x1x2x1x4xf32>
  %2 = linalg.init_tensor [%1, 2, 4] : tensor<?x2x4xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
     iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0, %arg1 : tensor<?x2x4xf32>, tensor<?x2x4xf32>)
    outs(%2 : tensor<?x2x4xf32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %4 = mulf %arg2, %arg3 : f32
    linalg.yield %4 : f32
  } -> tensor<?x2x4xf32>
  return %3 : tensor<?x2x4xf32>
}
//      CHECK: func @unit_dim_reshape_expansion_full
//  CHECK-DAG:   linalg.tensor_collapse_shape
//  CHECK-DAG:   linalg.init_tensor
//      CHECK:   linalg.generic
// CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<?x2x4xf32>, tensor<?x2x4xf32>)

//         FOLDUNITDIM: func @unit_dim_reshape_expansion_full
//    FOLDUNITDIM-SAME:   %[[ARG0:.+]]: tensor<1x?x1x2x1x4xf32>
//    FOLDUNITDIM-SAME:   %[[ARG1:.+]]: tensor<?x2x4xf32>
//     FOLDUNITDIM-DAG:   %[[RESHAPE:.+]] = linalg.tensor_expand_shape %[[ARG1]]
//         FOLDUNITDIM:   linalg.generic
//    FOLDUNITDIM-SAME:     ins(%[[ARG0]], %[[RESHAPE]] : tensor<1x?x1x2x1x4xf32>, tensor<1x?x1x2x1x4xf32>)
//    FOLDUNITDIM-SAME:     outs(%{{.+}} : tensor<1x?x1x2x1x4xf32>)


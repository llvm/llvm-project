// RUN: mlir-opt -split-input-file -linalg-fold-reshape-ops-by-linearization %s | FileCheck %s


// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?xf32>,
                                         %arg1 : tensor<?x?x4x?xf32>) ->
                                         tensor<?x?x4x?xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(i, j, k, l) -> (i)>,
                                    affine_map<(i, j, k, l) -> (j, k)>,
                                    affine_map<(i, j, k, l) -> (l)>] :
    tensor<?x?x?xf32> into tensor<?x?x4x?xf32>
  %1 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?x4x?xf32>, tensor<?x?x4x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x4x?xf32>
  return %1 : tensor<?x?x4x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_producer_fusion
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP1]]]
//   CHECK-NOT: linalg.generic


// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?x4x5xf32>,
                                         %arg1 : tensor<?x?x4x5xf32>) ->
                                         tensor<?x?xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?x4x5xf32>, tensor<?x?x4x5xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x4x5xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x4x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_consumer_fusion
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.generic

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @generic_op_reshape_consumer_nofusion(%arg0 : tensor<?x?x?x5xf32>,
                                           %arg1 : tensor<?x?x?x5xf32>) ->
                                           tensor<?x?xf32>
{
  %0 = linalg.generic {
     indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?x?x5xf32>, tensor<?x?x?x5xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):       // no predecessors
      %1 = mulf %arg3, %arg4 : f32
      linalg.yield %1 : f32
  } -> tensor<?x?x?x5xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x?x5xf32> into tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-LABEL: func @generic_op_reshape_consumer_nofusion
//       CHECK: linalg.tensor_reshape

// -----


// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @indexed_generic_op_reshape_producer_fusion(%arg0 : tensor<?x?x?xi32>)
  -> tensor<?x?x4x?xi32> {
  %0 = linalg.tensor_reshape %arg0 [affine_map<(i, j, k, l) -> (i)>,
                                    affine_map<(i, j, k, l) -> (j, k)>,
                                    affine_map<(i, j, k, l) -> (l)>] :
    tensor<?x?x?xi32> into tensor<?x?x4x?xi32>
  %1 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%0 : tensor<?x?x4x?xi32>) {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = addi %arg6, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<?x?x4x?xi32>
  return %1 : tensor<?x?x4x?xi32>
}

// CHECK-LABEL: func @indexed_generic_op_reshape_producer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.indexed_generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 * 20 + d2 * 5 + d3)>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @indexed_generic_op_reshape_consumer_fusion(%arg0 : tensor<?x?x4x5xi32>)
  -> tensor<?x?xi32> {
  %0 = linalg.indexed_generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"] }
    ins(%arg0 : tensor<?x?x4x5xi32>) {
  ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: i32):       // no predecessors
    %2 = index_cast %arg2 : index to i32
    %3 = addi %arg6, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<?x?x4x5xi32>
  %1 = linalg.tensor_reshape %0 [affine_map<(i, j, k, l) -> (i)>,
                                 affine_map<(i, j, k, l) -> (j, k, l)>] :
    tensor<?x?x4x5xi32> into tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: func @indexed_generic_op_reshape_consumer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.indexed_generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1 + d2 * 7)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_021_permultation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<3x7x5xf32> {
  %0 = linalg.tensor_reshape %arg0 [#map0, #map1] : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<3x5x7xf32>) {
    ^bb0(%arg2: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<3x7x5xf32>
    return %1 : tensor<3x7x5xf32>
}

// CHECK-LABEL: func @generic_op_021_permultation_reshape_producer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d2, d0 * 7 + d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_120_permultation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<5x7x3xf32> {
  %0 = linalg.tensor_reshape %arg0 [#map0, #map1] : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<3x5x7xf32>) {
    ^bb0(%arg2: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<5x7x3xf32>
    return %1 : tensor<5x7x3xf32>
}

// CHECK-LABEL: func @generic_op_120_permultation_reshape_producer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d1, d0 * 7 + d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_102_permultation_reshape_producer_fusion(%arg0 : tensor<3x35xf32>) -> tensor<5x3x7xf32> {
  %0 = linalg.tensor_reshape %arg0 [#map0, #map1] : tensor<3x35xf32> into tensor<3x5x7xf32>
  %1 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<3x5x7xf32>) {
    ^bb0(%arg2: f32):  // no predecessors
      linalg.yield %arg2 : f32
    } -> tensor<5x3x7xf32>
    return %1 : tensor<5x3x7xf32>
}

// CHECK-LABEL: func @generic_op_102_permultation_reshape_producer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d0 * 7 + d2)>


#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
func @generic_op_102_permultation_reshape_consumer_fusion(%arg0 : tensor<3x5x7xf32>) -> tensor<5x21xf32> {
  %0 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<3x5x7xf32>) {
    ^bb0(%arg2: f32):  // no predecessors
      linalg.yield %arg2 : f32
  } -> tensor<5x3x7xf32>
  %1 = linalg.tensor_reshape %0 [#map2, #map3] : tensor<5x3x7xf32> into tensor<5x21xf32>
  return %1 : tensor<5x21xf32>
}

// CHECK-LABEL: func @generic_op_102_permultation_reshape_consumer_fusion
//   CHECK-NOT: linalg.tensor_reshape
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//   CHECK-NOT: linalg.tensor_reshape

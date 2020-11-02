// RUN: mlir-opt %s -linalg-fold-unit-extent-dims -split-input-file | FileCheck %s

#accesses = [
  affine_map<(i, j, k, l, m) -> (i, k, m)>,
  affine_map<(i, j, k, l, m) -> (i, k, j, l, m)>
]

#trait = {
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  library_call = "some_external_func"
}

func @drop_one_trip_loops(%arg0 : tensor<?x1x?xf32>) -> tensor<?x1x?x1x?xf32>
{
  %0 = linalg.generic #trait
    ins(%arg0 : tensor<?x1x?xf32>) {
       ^bb0(%arg1 : f32) :
         linalg.yield %arg1 : f32
       } -> tensor<?x1x?x1x?xf32>
  return %0 : tensor<?x1x?x1x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2)>
//   CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//   CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
//   CHECK-DAG: #[[$MAP6:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-LABEL: func @drop_one_trip_loops
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[$MAP0]], #[[$MAP1]]]
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP2]], #[[$MAP3]]]
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[$MAP4]], #[[$MAP5]], #[[$MAP6]]]

// -----

#accesses = [
  affine_map<(i, j, k, l, m) -> (i, k, m)>,
  affine_map<(i, j, k, l, m) -> (i, k, j, l, m)>
]

#trait = {
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  library_call = "some_external_func"
}

func @drop_one_trip_loops_indexed_generic
  (%arg0 : tensor<?x1x?xi32>) -> tensor<?x1x?x1x?xi32>
{
  %0 = linalg.indexed_generic #trait
    ins(%arg0 : tensor<?x1x?xi32>) {
       ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
            %arg5 : index, %arg6 : i32) :
	 %1 = addi %arg1, %arg2 : index
	 %2 = addi %1, %arg3 : index
	 %3 = addi %2, %arg4 : index
	 %4 = addi %3, %arg5 : index
	 %5 = index_cast %4 : index to i32
	 %6 = addi %5, %arg6 : i32
         linalg.yield %6 : i32
       } -> tensor<?x1x?x1x?xi32>
  return %0 : tensor<?x1x?x1x?xi32>
}
// CHECK-LABEL: func @drop_one_trip_loops_indexed_generic
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{.+}}(
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index, %[[ARG2:[a-zA-Z0-9]+]]: index
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index, %[[ARG4:[a-zA-Z0-9]+]]: i32)
//       CHECK:     %[[T3:.+]] = addi %[[ARG1]], %[[ARG2]]
//       CHECK:     %[[T4:.+]] = addi %[[T3]], %[[ARG3]]
//       CHECK:     %[[T5:.+]] = index_cast %[[T4]] : index to i32
//       CHECK:     %[[T6:.+]] = addi %[[T5]], %[[ARG4]] : i32
//       CHECK:     linalg.yield %[[T6]] : i32

// -----

#map0 = affine_map<(i, j) -> (i, j)>
#access = [#map0, #map0]
#trait = {
  iterator_types = ["parallel", "parallel"],
  indexing_maps = #access,
  library_call = "some_external_func"
}

func @drop_all_loops(%arg0 : tensor<1x1xf32>) -> tensor<1x1xf32>
{
  %0 = linalg.generic #trait
    ins(%arg0 : tensor<1x1xf32>) {
       ^bb0(%arg1: f32) :
         linalg.yield %arg1 : f32
       } -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @drop_all_loops
//       CHECK:   linalg.tensor_reshape %{{.*}} []
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//  CHECK-SAME:     iterator_types = []

// -----

#map0 = affine_map<(i, j) -> (i, j)>
#access = [#map0, #map0]
#trait = {
  iterator_types = ["parallel", "parallel"],
  indexing_maps = #access,
  library_call = "some_external_func"
}

func @drop_all_loops_indexed_generic
  (%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32>
{
  %0 = linalg.indexed_generic #trait
    ins(%arg0 : tensor<1x1xi32>) {
       ^bb0(%arg1 : index, %arg2 : index, %arg3: i32) :
         %1 = addi %arg1, %arg2 : index
	 %2 = index_cast %1 : index to i32
	 %3 = addi %2, %arg3 : i32
         linalg.yield %3 : i32
       } -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: func @drop_all_loops_indexed_generic
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{.+}}(%[[ARG1:.+]]: i32)
//       CHECK:     linalg.yield %[[ARG1]] : i32

// -----

#accesses = [
  affine_map<(d0) -> (0, d0)>,
  affine_map<(d0) -> (d0)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel"],
  library_call = "some_external_fn"
}

func @leading_dim_1_canonicalization(%arg0: tensor<1x5xf32>) -> tensor<5xf32> {
  %0 = linalg.generic #trait
    ins(%arg0 : tensor<1x5xf32>) {
  ^bb0(%arg2: f32):     // no predecessors
    linalg.yield %arg2 : f32
  } -> tensor<5xf32>
  return %0 : tensor<5xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @leading_dim_1_canonicalization
//       CHECK:   linalg.tensor_reshape %{{.*}} [#[[$MAP0]]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP1]], #[[$MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel"]

// -----

#accesses = [
  affine_map<(d0, d1) -> (0, d1)>,
  affine_map<(d0, d1) -> (d0, 0)>,
  affine_map<(d0, d1) -> (d0, d1)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"],
  library_call = "some_external_fn"
}

func @broadcast_test(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<5x5xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] :
       tensor<5xf32> into tensor<1x5xf32>
  %1 = linalg.tensor_reshape %arg1 [affine_map<(d0, d1) -> (d0, d1)>] :
       tensor<5xf32> into tensor<5x1xf32>
  %2 = linalg.generic #trait
    ins(%0, %1 : tensor<1x5xf32>, tensor<5x1xf32>) {
       ^bb0(%arg2: f32, %arg3: f32):
         %3 = addf %arg2, %arg3 : f32
         linalg.yield %3 : f32
       } -> tensor<5x5xf32>
  return %2 : tensor<5x5xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d1)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
//   CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @broadcast_test
//   CHECK-NOT:   linalg.tensor_reshape
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//   CHECK-NOT:   linalg.tensor_reshape

// -----

#accesses = [
  affine_map<(d0, d1) -> (0, 0)>,
  affine_map<(d0, d1) -> (d0, d1)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"],
  library_call = "some_external_fn"
}

func @broadcast_scalar(%arg0 : tensor<1x1xf32>) -> tensor<?x?xf32>
{
   %0 = linalg.generic #trait
    ins(%arg0 : tensor<1x1xf32>) {
      ^bb0(%arg1 : f32):
        linalg.yield %arg1 : f32
   } -> tensor<?x?xf32>
   return %0 : tensor<?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> ()>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @broadcast_scalar
//  CHECK-SAME:   %[[ARG0:.*]]: tensor<1x1xf32>
//       CHECK:   %[[A:.*]] = linalg.tensor_reshape %[[ARG0]] []
//  CHECK-SAME:     tensor<1x1xf32> into tensor<f32>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     %[[A]]

// -----

//       CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//       CHECK: func @fold_reshape
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]]]
//  CHECK-SAME:   tensor<2048xf32> into tensor<4x512xf32>
func @fold_reshape(%arg0 : tensor<2048xf32>) -> tensor<4x512xf32>
{
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
    : tensor<2048xf32> into tensor<1x4x1x512xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
     affine_map<(d0, d1, d2, d3) -> (d3)>]
    : tensor<1x4x1x512xf32> into tensor<4x512xf32>
  return %1 : tensor<4x512xf32>
}

// -----

//       CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//       CHECK: func @fold_reshape
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]]]
//  CHECK-SAME:   tensor<4x512xf32> into tensor<2048xf32>
func @fold_reshape(%arg0 : tensor<4x512xf32>) -> tensor<2048xf32>
{
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
     affine_map<(d0, d1, d2, d3) -> (d3)>]
    : tensor<4x512xf32> into tensor<1x4x1x512xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]
    : tensor<1x4x1x512xf32> into tensor<2048xf32>
  return %1 : tensor<2048xf32>
}

// -----

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2)>
//       CHECK: func @fold_reshape
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:   tensor<2048x1xf32> into tensor<4x512x1xf32>
func @fold_reshape(%arg0 : tensor<2048x1xf32>) -> tensor<4x512x1xf32>
{
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d4)>]
    : tensor<2048x1xf32> into tensor<1x4x1x512x1xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d3)>,
     affine_map<(d0, d1, d2, d3, d4) -> (d4)>]
    : tensor<1x4x1x512x1xf32> into tensor<4x512x1xf32>
  return %1 : tensor<4x512x1xf32>
}

// -----

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4)>
//       CHECK: func @fold_reshape
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
//  CHECK-SAME:   tensor<2048x1x2048xf32> into tensor<4x512x1x512x4xf32>
func @fold_reshape(%arg0 : tensor<2048x1x2048xf32>) -> tensor<4x512x1x512x4xf32>
{
  %0 = linalg.tensor_reshape %arg0
    [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d6, d7, d8)>]
    : tensor<2048x1x2048xf32> into tensor<1x4x1x512x1x1x512x1x4xf32>
  %1 = linalg.tensor_reshape %0
    [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d3, d4)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d5)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d6, d7)>,
     affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d8)>]
    : tensor<1x4x1x512x1x1x512x1x4xf32> into tensor<4x512x1x512x4xf32>
  return %1 : tensor<4x512x1x512x4xf32>
}

// -----

//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//       CHECK: func @fold_reshape
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]]
//  CHECK-SAME:   tensor<2xf32> into tensor<2x1xf32>
func @fold_reshape(%arg0: tensor<2xf32>) -> tensor<2x1xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1, d2) -> (d0, d1, d2)>] : tensor<2xf32> into tensor<2x1x1xf32>
  %1 = linalg.tensor_reshape %0
  [affine_map<(d0, d1, d2) -> (d0)>,
   affine_map<(d0, d1, d2) -> (d1, d2)>
  ] : tensor<2x1x1xf32> into tensor<2x1xf32>
  return %1 : tensor<2x1xf32>
}

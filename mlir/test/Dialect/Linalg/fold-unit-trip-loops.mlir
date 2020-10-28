// RUN: mlir-opt %s -linalg-fold-unit-extent-dims="fold-one-trip-loops-only" -split-input-file | FileCheck %s

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
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d1, 0, d2)>
// CHECK-LABEL: func @drop_one_trip_loops
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]

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
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<() -> (0, 0)>
// CHECK-LABEL: func @drop_all_loops
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

func @drop_all_loops(%arg0 : memref<1x1xf32>, %arg1 : memref<1x1xf32>)
{
  linalg.generic #trait
     ins(%arg0 : memref<1x1xf32>)
    outs(%arg1 : memref<1x1xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32) :
      linalg.yield %arg2 : f32
    }
  return
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<() -> (0, 0)>
// CHECK-LABEL: func @drop_all_loops
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//  CHECK-SAME:     iterator_types = []

// -----

#accesses = [
  affine_map<(d0, d1) -> (d0, d1)>,
  affine_map<(d0, d1) -> (d1)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"],
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
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (0, d0)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @leading_dim_1_canonicalization
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel"]

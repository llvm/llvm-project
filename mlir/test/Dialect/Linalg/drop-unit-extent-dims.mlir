// RUN: mlir-opt %s -split-input-file -linalg-fold-unit-extent-dims | FileCheck %s

#accesses = [
  affine_map<(i, j, k, l, m) -> (i, k, m)>,
  affine_map<(i, j, k, l, m) -> (i, k, j, l, m)>
]

#trait = {
  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"],
  indexing_maps = #accesses,
  library_call = "some_external_func"
}

func @drop_one_trip_loops(%arg0 : tensor<?x1x?xf32>, %shape: tensor<?x1x?x1x?xf32>) -> tensor<?x1x?x1x?xf32> {
  %0 = linalg.generic #trait
     ins(%arg0 : tensor<?x1x?xf32>)
    outs(%shape : tensor<?x1x?x1x?xf32>) {
       ^bb0(%arg2 : f32, %arg3 : f32) :
         linalg.yield %arg2 : f32
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
  (%arg0 : tensor<?x1x?xi32>, %shape: tensor<?x1x?x1x?xi32>) -> tensor<?x1x?x1x?xi32>
{
  %0 = linalg.indexed_generic #trait
     ins(%arg0 : tensor<?x1x?xi32>)
    outs(%shape: tensor<?x1x?x1x?xi32>) {
       ^bb0(%arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
            %arg5 : index, %arg6 : i32, %arg7 : i32) :
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
//  CHECK-SAME:     %[[ARG3:[a-zA-Z0-9]+]]: index, %[[ARG4:[a-zA-Z0-9]+]]: i32, %{{.*}}: i32)
//       CHECK:     %[[T3:.+]] = addi %[[ARG1]], %[[ARG2]]
//       CHECK:     %[[T4:.+]] = addi %[[T3]], %[[ARG3]]
//       CHECK:     %[[T5:.+]] = index_cast %[[T4]] : index to i32
//       CHECK:     %[[T6:.+]] = addi %[[T5]], %[[ARG4]] : i32
//       CHECK:     linalg.yield %[[T6]] : i32

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

func @drop_one_trip_loops_indexed
  (%arg0 : tensor<?x1x?xi32>, %shape: tensor<?x1x?x1x?xi32>) -> tensor<?x1x?x1x?xi32>
{
  %0 = linalg.generic #trait
     ins(%arg0 : tensor<?x1x?xi32>)
    outs(%shape: tensor<?x1x?x1x?xi32>) {
       ^bb0(%arg6 : i32, %arg7 : i32) :
         %idx0 = linalg.index 0 : index
         %idx1 = linalg.index 1 : index
         %idx2 = linalg.index 2 : index
         %idx3 = linalg.index 3 : index
         %idx4 = linalg.index 4 : index
         %1 = addi %idx0, %idx1 : index
         %2 = subi %1, %idx2 : index
         %3 = subi %2, %idx3 : index
         %4 = addi %3, %idx4 : index
         %5 = index_cast %4 : index to i32
         %6 = addi %5, %arg6 : i32
         linalg.yield %6 : i32
       } -> tensor<?x1x?x1x?xi32>
  return %0 : tensor<?x1x?x1x?xi32>
}
// The subtractions disappear the access map of the output tensor maps its unit
// dimensions 1 and 3 to the index dimensions 2 and 3.
// CHECK-LABEL: func @drop_one_trip_loops_indexed
//       CHECK:   linalg.generic
//       CHECK:   ^{{.+}}(
//  CHECK-SAME:     %[[ARG4:[a-zA-Z0-9]+]]: i32, %{{.*}}: i32)
//       CHECK:     %[[IDX0:.+]] = linalg.index 0 : index
//       CHECK:     %[[IDX1:.+]] = linalg.index 1 : index
//       CHECK:     %[[IDX2:.+]] = linalg.index 2 : index
//       CHECK:     %[[T3:.+]] = addi %[[IDX0]], %[[IDX1]]
//       CHECK:     %[[T4:.+]] = addi %[[T3]], %[[IDX2]]
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
     ins(%arg0 : tensor<1x1xf32>)
    outs(%arg0 : tensor<1x1xf32>) {
       ^bb0(%arg1: f32, %arg2: f32) :
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
  (%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32>{
  %0 = linalg.indexed_generic #trait
     ins(%arg0 : tensor<1x1xi32>)
    outs(%arg0 : tensor<1x1xi32>) {
       ^bb0(%arg1 : index, %arg2 : index, %arg3: i32, %arg4: i32) :
         %1 = addi %arg1, %arg2 : index
         %2 = index_cast %1 : index to i32
         %3 = addi %2, %arg3 : i32
         linalg.yield %3 : i32
       } -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: func @drop_all_loops_indexed_generic
//       CHECK:   linalg.indexed_generic
//       CHECK:   ^{{.+}}(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
//       CHECK:     linalg.yield %[[ARG1]] : i32

// -----

#map0 = affine_map<(i, j) -> (i, j)>
#access = [#map0, #map0]
#trait = {
  iterator_types = ["parallel", "parallel"],
  indexing_maps = #access,
  library_call = "some_external_func"
}

func @drop_all_loops_indexed
  (%arg0 : tensor<1x1xi32>) -> tensor<1x1xi32>{
  %0 = linalg.generic #trait
     ins(%arg0 : tensor<1x1xi32>)
    outs(%arg0 : tensor<1x1xi32>) {
       ^bb0(%arg3: i32, %arg4: i32) :
         %idx0 = linalg.index 0 : index
         %idx1 = linalg.index 1 : index
         %1 = addi %idx0, %idx1 : index
         %2 = index_cast %1 : index to i32
         %3 = addi %2, %arg3 : i32
         linalg.yield %3 : i32
       } -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// CHECK-LABEL: func @drop_all_loops_indexed
//       CHECK:   linalg.generic
//       CHECK:   ^{{.+}}(%[[ARG1:.+]]: i32, %[[ARG2:.+]]: i32)
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

func @leading_dim_1_canonicalization(%arg0: tensor<1x5xf32>, %shape: tensor<5xf32>) -> tensor<5xf32> {
  %0 = linalg.generic #trait
     ins(%arg0 : tensor<1x5xf32>)
    outs(%shape : tensor<5xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):     // no predecessors
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

func @broadcast_test(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>, %shape : tensor<5x5xf32>) -> tensor<5x5xf32>
{
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1) -> (d0, d1)>] :
       tensor<5xf32> into tensor<1x5xf32>
  %1 = linalg.tensor_reshape %arg1 [affine_map<(d0, d1) -> (d0, d1)>] :
       tensor<5xf32> into tensor<5x1xf32>
  %2 = linalg.generic #trait
     ins(%0, %1 : tensor<1x5xf32>, tensor<5x1xf32>)
    outs(%shape : tensor<5x5xf32>) {
       ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
         %3 = addf %arg3, %arg4 : f32
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

func @broadcast_scalar(%arg0 : tensor<1x1xf32>, %shape : tensor<?x?xf32>) -> tensor<?x?xf32>
{
   %0 = linalg.generic #trait
     ins(%arg0 : tensor<1x1xf32>)
    outs(%shape : tensor<?x?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32):
        linalg.yield %arg2 : f32
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

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
func @fold_unit_dim_tensor_reshape_op(%arg0 : tensor<5xf32>) -> tensor<2x5xf32>
{
  %1 = linalg.init_tensor [1, 2, 5] : tensor<1x2x5xf32>
  %2 = linalg.generic {i64, indexing_maps = [#map1, #map0],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<5xf32>) outs(%1 : tensor<1x2x5xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x2x5xf32>
  %3 = linalg.tensor_reshape %2 [#map3, #map4]
    : tensor<1x2x5xf32> into tensor<2x5xf32>
  return %3 : tensor<2x5xf32>
}
// CHECK-LABEL: func @fold_unit_dim_tensor_reshape_op
//       CHECK:   %[[RESULT:.+]] = linalg.generic
//       CHECK:   return %[[RESULT]]

// -----

func @fold_unit_dim_for_init_tensor(%input: tensor<1x1000xf32>) -> tensor<1xf32> {
  %cst = constant 0.0 : f32
  %init = linalg.init_tensor [1] : tensor<1xf32>
  %fill = linalg.fill(%init, %cst) : tensor<1xf32>, f32 -> tensor<1xf32>
  %add = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
    ins(%input : tensor<1x1000xf32>)outs(%fill : tensor<1xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %1823 = addf %arg1, %arg2 : f32
    linalg.yield %1823 : f32
  } -> tensor<1xf32>
  return %add : tensor<1xf32>
}


//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> ()>

//       CHECK: func @fold_unit_dim_for_init_tensor


//       CHECK: %[[INPUT_RESHAPE:.+]] = linalg.tensor_reshape %{{.+}} [#[[MAP0]]] : tensor<1x1000xf32> into tensor<1000xf32>
//       CHECK: %[[INIT:.+]] = linalg.init_tensor [] : tensor<f32>
//       CHECK: %[[FILL:.+]] = linalg.fill(%[[INIT]], %cst) : tensor<f32>, f32 -> tensor<f32>
//       CHECK: %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP1]], #[[MAP2]]]
//  CHECK-SAME:     iterator_types = ["reduction"]
//  CHECK-SAME:   ins(%[[INPUT_RESHAPE]] : tensor<1000xf32>)
//  CHECK-SAME:   outs(%[[FILL]] : tensor<f32>)
//       CHECK: %[[GENERIC_RESHAPE:.+]] = linalg.tensor_reshape %[[GENERIC]] [] : tensor<f32> into tensor<1xf32>
//       CHECK: return %[[GENERIC_RESHAPE:.+]] : tensor<1xf32>


// -----

func @fold_subtensor(
    %arg0 : tensor<1x?x?x1x?x1x1xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index)
    -> tensor<1x?x?x1x?x1x1xf32> {
  %0 = subtensor %arg0[0, %arg1, %arg2, 0, %arg3, 0, 0]
                      [1, %arg4, %arg5, 1, %arg6, 1, 1] [1, 1, 1, 1, 1, 1, 1] :
      tensor<1x?x?x1x?x1x1xf32> to tensor<1x?x?x1x?x1x1xf32>
  return %0 : tensor<1x?x?x1x?x1x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
//      CHECK: func @fold_subtensor
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x?x1x?x1x1xf32>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG3:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG4:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG5:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG6:[a-z0-9]+]]: index
//      CHECK:   %[[SRC_RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
//      CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[SRC_RESHAPE]]
// CHECK-SAME:       [%[[ARG1]], %[[ARG2]], %[[ARG3]]]
// CHECK-SAME:       [%[[ARG4]], %[[ARG5]], %[[ARG6]]]
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[SUBTENSOR]]
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @no_fold_subtensor(
    %arg0 : tensor<1x?x?x?x?x1x1xf32>, %arg1 : index, %arg2 : index,
    %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index)
    -> tensor<1x?x?x1x?x1x1xf32> {
  %0 = subtensor %arg0[%arg1, 0, %arg2, 0, 0, %arg3, 0]
                      [1, %arg4, %arg5, 1, %arg6, 1, 1] [1, 1, 1, 1, 1, 1, 1] :
      tensor<1x?x?x?x?x1x1xf32> to tensor<1x?x?x1x?x1x1xf32>
  return %0 : tensor<1x?x?x1x?x1x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4)>
//  CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6)>
//      CHECK: func @no_fold_subtensor
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x?x?x?x1x1xf32>
// CHECK-SAME:   %[[ARG1:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG2:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG3:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG4:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG5:[a-z0-9]+]]: index
// CHECK-SAME:   %[[ARG6:[a-z0-9]+]]: index
//      CHECK:   %[[SRC_RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]]
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]], #[[MAP5]]]
//      CHECK:   %[[SUBTENSOR:.+]] = subtensor %[[SRC_RESHAPE]]
// CHECK-SAME:       [%[[ARG1]], 0, %[[ARG2]], 0, 0, %[[ARG3]]]
// CHECK-SAME:       [1, %[[ARG4]], %[[ARG5]], 1, %[[ARG6]], 1]
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[SUBTENSOR]]
// CHECK-SAME:       [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]], #[[MAP5]]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @unit_dim_for_reduction(%arg0: tensor<1x?x1x?xf32>) -> tensor<1x?xf32> {
  %cst = constant 1.000000e+00 : f32
  %c3 = constant 3 : index
  %0 = memref.dim %arg0, %c3 : tensor<1x?x1x?xf32>
  %1 = linalg.init_tensor [1, %0] : tensor<1x?xf32>
  %2 = linalg.fill(%1, %cst) : tensor<1x?xf32>, f32 -> tensor<1x?xf32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0 : tensor<1x?x1x?xf32>)
    outs(%2 : tensor<1x?xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %4 = addf %arg1, %arg2 : f32
    linalg.yield %4 : f32
  } -> tensor<1x?xf32>
  return %3 : tensor<1x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x1x?xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] [#[[MAP0]], #[[MAP1]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] [#[[MAP2]]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @unit_dim_for_reduction_keep_one(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
  %cst = constant 1.000000e+00 : f32
  %c3 = constant 3 : index
  %1 = linalg.init_tensor [1, 1] : tensor<1x1xf32>
  %2 = linalg.fill(%1, %cst) : tensor<1x1xf32>, f32 -> tensor<1x1xf32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0 : tensor<1x?x1x1xf32>)
    outs(%2 : tensor<1x1xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %4 = addf %arg1, %arg2 : f32
    linalg.yield %4 : f32
  } -> tensor<1x1xf32>
  return %3 : tensor<1x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction_keep_one
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x1x1xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] [#[[MAP0]], #[[MAP1]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [1] : tensor<1xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x1xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<1xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] [#[[MAP2]]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @unit_dim_for_reduction_inner(%arg0: tensor<?x1x?x1xf32>) -> tensor<?x1xf32> {
  %cst = constant 1.000000e+00 : f32
  %c2 = constant 2 : index
  %0 = memref.dim %arg0, %c2 : tensor<?x1x?x1xf32>
  %1 = linalg.init_tensor [%0, 1] : tensor<?x1xf32>
  %2 = linalg.fill(%1, %cst) : tensor<?x1xf32>, f32 -> tensor<?x1xf32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                     affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0 : tensor<?x1x?x1xf32>)
    outs(%2 : tensor<?x1xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %4 = addf %arg1, %arg2 : f32
    linalg.yield %4 : f32
  } -> tensor<?x1xf32>
  return %3 : tensor<?x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction_inner
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x1x?x1xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] [#[[MAP0]], #[[MAP1]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] [#[[MAP2]]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @no_fold_reshape_empty_expr(%arg0: tensor<3x2x2xf32>) -> tensor<12x1xf32> {
  %0 = linalg.tensor_reshape %arg0 [affine_map<(d0, d1, d2, d3) -> (d0)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>] : tensor<3x2x2xf32> into tensor<3x2x2x1xf32>
  %1 = linalg.tensor_reshape %0 [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d3)>] : tensor<3x2x2x1xf32> into tensor<12x1xf32>
  return %1 : tensor<12x1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
//      CHECK: func @no_fold_reshape_empty_expr
// CHECK-SAME:    %[[ARG0:.+]]: tensor<3x2x2xf32>
//      CHECK:    %[[RARG0:.+]] = linalg.tensor_reshape %[[ARG0:.+]] [#[[MAP0]], #[[MAP1]], #[[MAP2]]
//      CHECK:    %[[RES:.+]] = linalg.tensor_reshape %[[RARG0:.+]] [#[[MAP3]], #[[MAP4]]]
//      CHECK:    return %[[RES:.+]] : tensor<12x1xf32>

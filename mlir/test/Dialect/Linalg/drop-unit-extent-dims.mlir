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
//   CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//   CHECK-DAG: #[[$MAP3:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @drop_one_trip_loops
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1], [2]]
//       CHECK: linalg.generic
//  CHECK-SAME:   indexing_maps = [#[[$MAP2]], #[[$MAP3]]]
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
//       CHECK: linalg.tensor_reshape %{{.*}} {{\[}}[0, 1], [2, 3], [4]]

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
//       CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
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
//   CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @leading_dim_1_canonicalization
//       CHECK:   linalg.tensor_reshape %{{.*}} {{\[}}[0, 1]]
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
  %0 = linalg.tensor_reshape %arg0 [[0, 1]] : tensor<5xf32> into tensor<1x5xf32>
  %1 = linalg.tensor_reshape %arg1 [[0, 1]] : tensor<5xf32> into tensor<5x1xf32>
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

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
func @fold_unit_dim_tensor_reshape_op(%arg0 : tensor<5xf32>) -> tensor<2x5xf32>
{
  %1 = linalg.init_tensor [1, 2, 5] : tensor<1x2x5xf32>
  %2 = linalg.generic {i64, indexing_maps = [#map1, #map0],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<5xf32>) outs(%1 : tensor<1x2x5xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x2x5xf32>
  %3 = linalg.tensor_reshape %2 [[0, 1], [2]]
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


//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0) -> ()>

//       CHECK: func @fold_unit_dim_for_init_tensor


//       CHECK: %[[INPUT_RESHAPE:.+]] = linalg.tensor_reshape %{{.+}} {{\[}}[0, 1]] : tensor<1x1000xf32> into tensor<1000xf32>
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
    %arg0 : tensor<1x?x?x1x?x1x1xf32>, %arg1 : tensor<1x?x?x?x?x1x1xf32>,
    %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : index,
    %arg6 : index, %arg7 : index) -> (tensor<1x?x?x1x?x1x1xf32>, tensor<1x?x?x1x?x1x1xf32>) {
  %0 = subtensor %arg0[0, %arg2, %arg3, 0, %arg4, 0, 0]
                      [1, %arg5, %arg6, 1, %arg7, 1, 1] [1, 1, 1, 1, 1, 1, 1] :
      tensor<1x?x?x1x?x1x1xf32> to tensor<1x?x?x1x?x1x1xf32>
  %1 = subtensor %arg1[%arg2, 0, %arg3, 0, 0, %arg4, 0]
                      [1, %arg5, %arg6, 1, %arg7, 1, 1] [1, 1, 1, 1, 1, 1, 1] :
      tensor<1x?x?x?x?x1x1xf32> to tensor<1x?x?x1x?x1x1xf32>
  return %0, %1 : tensor<1x?x?x1x?x1x1xf32>, tensor<1x?x?x1x?x1x1xf32>
}
//      CHECK: func @fold_subtensor
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x?x1x?x1x1xf32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<1x?x?x?x?x1x1xf32>
//      CHECK:   %[[SUBTENSOR1:.+]] = subtensor %[[ARG0]]
// CHECK-SAME:       to tensor<?x?x?xf32>
//      CHECK:   %[[RESULT1:.+]] = linalg.tensor_reshape %[[SUBTENSOR1]]
// CHECK-SAME:       [0, 1], [2], [3, 4, 5, 6]
//      CHECK:   %[[SUBTENSOR2:.+]] = subtensor %[[ARG1]]
// CHECK-SAME:       to tensor<?x?x?xf32>
//      CHECK:   %[[RESULT2:.+]] = linalg.tensor_reshape %[[SUBTENSOR2]]
// CHECK-SAME:       [0, 1], [2], [3, 4, 5, 6]
//      CHECK:   return %[[RESULT1]], %[[RESULT2]]

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
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x1x?xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] {{\[}}[0, 1]]
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
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction_keep_one
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x?x1x1xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [1] : tensor<1xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x1xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<1xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] {{\[}}[0, 1]]
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
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: func @unit_dim_for_reduction_inner
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x1x?x1xf32>
//  CHECK-DAG:   %[[RESHAPE:.+]] = linalg.tensor_reshape %[[ARG0]] {{\[}}[0, 1], [2, 3]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%{{.+}}] : tensor<?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill(%[[INIT]], %{{.+}})
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP2]], #[[MAP3]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[RESHAPE]] : tensor<?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?xf32>)
//      CHECK:   %[[RESULT_RESHAPE:.+]] = linalg.tensor_reshape %[[RESULT]] {{\[}}[0, 1]]
//      CHECK:   return %[[RESULT_RESHAPE]]

// -----

func @subtensor_unit_dims(%arg0: tensor<1x3xf32>) -> tensor<1x1xf32> {
  %0 = subtensor %arg0[0, 2] [1, 1] [1, 1] : tensor<1x3xf32> to tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func @subtensor_unit_dims
//       CHECK:   %[[SUBTENSOR:.+]] = subtensor
//  CHECK-SAME:     tensor<1x3xf32> to tensor<f32>
//       CHECK:   %[[RESULT:.+]] = linalg.tensor_reshape %[[SUBTENSOR]] []
//       CHECK:   return %[[RESULT]]

// -----

func @subtensor_insert_unit_dims(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x3xf32> {
  %0 = subtensor_insert %arg1 into %arg0[0, 2] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK-LABEL: func @subtensor_insert_unit_dims
//       CHECK:   %[[RESHAPE:.+]] = linalg.tensor_reshape %{{.+}} []
//       CHECK:   %[[RESULT:.+]] = subtensor_insert %[[RESHAPE]]
//  CHECK-SAME:     tensor<f32> into tensor<1x3xf32>
//       CHECK:   return %[[RESULT]]

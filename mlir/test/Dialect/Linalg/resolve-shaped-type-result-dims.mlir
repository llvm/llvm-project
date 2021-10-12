// RUN: mlir-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func @init_tensor_static_dim() -> (index, index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %0 = linalg.init_tensor [4, 5, %c6] : tensor<4x5x?xf32>
  %1 = tensor.dim %0, %c2 : tensor<4x5x?xf32>
  %2 = tensor.dim %0, %c0 : tensor<4x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_static_dim
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
//      CHECK:   return %[[C6]], %[[C4]]

// -----

func @init_tensor_dynamic_dim(%arg0 : index) -> (index) {
  %c2 = arith.constant 2 : index
  %0 = linalg.init_tensor [4, 5, %arg0] : tensor<4x5x?xf32>
  %1 = tensor.dim %0, %c2 : tensor<4x5x?xf32>
  return %1 : index
}
//      CHECK: func @init_tensor_dynamic_dim
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]]

// -----

func @init_tensor_dynamic_dim2(%arg0 : index, %arg1 : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %1, %2 : index, index
}
//      CHECK: func @init_tensor_dynamic_dim2
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG0]], %[[ARG1]]

// -----

func @remove_dim_result_uses
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d2, d1)>,
                      affine_map<(d0, d1, d2) -> (d0 + d1, d1 - d0)>],
     iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
  %3 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %4 = tensor.dim %0, %c1 : tensor<?x?xf32>
  return %3, %4 : index, index
}
//       CHECK: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s1 - s0)>
//       CHECK: func @remove_dim_result_uses
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T2:.+]] = affine.apply #[[MAP0]]()[%[[T0]], %[[T1]]]
//   CHECK-DAG:   %[[T3:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T4:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[T5:.+]] = affine.apply #[[MAP1]]()[%[[T3]], %[[T4]]]
//       CHECK:   return %[[T2]], %[[T5]]

// -----

func @remove_dim_result_uses_outs
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3: f32) :
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = tensor.dim %1, %c1 : tensor<?x?xf32>
  return %2 : index
}
//      CHECK: func @remove_dim_result_uses_outs
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   return %[[ARG1]]

// -----

func @remove_dim_result_uses_sequence
  (%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
   %arg2 : tensor<?x?xf32>) -> (index, index, index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32>
  %3 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>,
                      affine_map<(d0, d1, d2) -> (d0, d2)>],
     iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg3 : f32, %arg4 : f32, %arg5 : f32):
      %4 = arith.mulf %arg3, %arg4 : f32
      %5 = arith.addf %4, %arg5 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  %6 = tensor.dim %3, %c0 : tensor<?x?xf32>
  %7 = tensor.dim %3, %c1 : tensor<?x?xf32>
  return %1, %2, %6, %7 : index, index, index, index
}
// CHECK-LABEL: func @remove_dim_result_uses_sequence
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[T0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[T1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//   CHECK-DAG:   %[[T2:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[T3:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   return %[[T0]], %[[T1]], %[[T2]], %[[T3]]

// -----

func @keep_result_dim_uses_sequence2
  (%arg0 : tensor<?xf32>, %arg1 : index) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %0 = linalg.init_tensor [%d0, %arg1] : tensor<?x?xf32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                      affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
    ^bb0(%arg2: f32, %arg3 : f32):
      linalg.yield %arg2 : f32
    } -> tensor<?x?xf32>
  %2 = tensor.dim %1, %c0 : tensor<?x?xf32>
  %3 = tensor.dim %1, %c1 : tensor<?x?xf32>
  return %2, %3 : index, index
}
//       CHECK: func @keep_result_dim_uses_sequence2
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[T0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   return %[[T0]], %[[ARG1]]

// -----

#map = affine_map<(d0) -> (d0)>

func @init_tensor_dim_of_linalg_result(%arg_0 : tensor<?xf32>,
    %arg_1: tensor<?xf32>) -> (index, index) {
  %0, %1 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%arg_0 : tensor<?xf32>)
    outs(%arg_0, %arg_1 : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%in: f32, %out_0: f32, %out_1: f32):
    linalg.yield %in, %in : f32, f32
  } -> (tensor<?xf32>, tensor<?xf32>)

  %c0 = arith.constant 0 : index
  %num_elem_0 = tensor.dim %0, %c0 : tensor<?xf32>

  %num_elem_1 = tensor.dim %1, %c0 : tensor<?xf32>
  return %num_elem_0, %num_elem_1 : index, index
}
//      CHECK: func @init_tensor_dim_of_linalg_result(
// CHECK-SAME:   %[[ARG_0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK-SAME:   %[[ARG_1:[a-zA-Z0-9_]+]]: tensor<?xf32>)
//      CHECK:   %[[R0:.+]] = tensor.dim %[[ARG_0]]
//      CHECK:   %[[R1:.+]] = tensor.dim %[[ARG_0]]
//      CHECK:   return %[[R0]], %[[R1]]

// -----

func @dim_reshape_expansion(%arg0 : tensor<6x5x?xf32>) -> (index, index, index)
{
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = linalg.tensor_expand_shape %arg0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  %1 = tensor.dim %0, %c1 : tensor<2x3x5x4x?x7xf32>
  %2 = tensor.dim %0, %c3 : tensor<2x3x5x4x?x7xf32>
  %3 = tensor.dim %0, %c4 : tensor<2x3x5x4x?x7xf32>
  return %1, %2, %3 : index, index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @dim_reshape_expansion
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<6x5x?xf32>
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C3]], %[[C4]], %[[D1]]

// -----

func @dim_reshape_collapse(%arg0 : tensor<2x3x5x4x?x7xf32>) -> (index, index)
{
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  %1 = tensor.dim %0, %c1 : tensor<6x5x?xf32>
  %2 = tensor.dim %0, %c2 : tensor<6x5x?xf32>
  return %1, %2 : index, index
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @dim_reshape_collapse
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<2x3x5x4x?x7xf32>
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C4]]
//      CHECK:   %[[D1:.+]] = affine.apply #[[MAP]]()[%[[D0]]]
//      CHECK:   return %[[C5]], %[[D1]]

// -----

func @dim_of_pad_op(%arg0 : tensor<2x?x?xf32>, %arg1 : index, %arg2 : index,
    %arg3: f32) -> (index, index, index)
{
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %c3 = arith.constant 3 : index
   %c4 = arith.constant 4 : index
   %c5 = arith.constant 5 : index
   %0 = linalg.pad_tensor %arg0 low[%c3, %arg1, %c4] high[7, %c5, %arg2] {
     ^bb0(%arg4: index, %arg5: index, %arg6: index):
       linalg.yield %arg3 : f32
   } : tensor<2x?x?xf32> to tensor<?x?x?xf32>
   %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
   %2 = tensor.dim %0, %c1 : tensor<?x?x?xf32>
   %3 = tensor.dim %0, %c2 : tensor<?x?x?xf32>
   return %1, %2, %3 : index, index, index
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s1 + s0 + 5)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s1 + s0 + 4)>
//      CHECK: func @dim_of_pad_op
// CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<2x?x?xf32>
// CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[A-Za-z0-9_]+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C12:.+]] = arith.constant 12 : index
//      CHECK:   %[[IN_DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[OUT_DIM1:.+]] = affine.apply #[[MAP0]]()[%[[ARG1]], %[[IN_DIM1]]]
//      CHECK:   %[[IN_DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//      CHECK:   %[[OUT_DIM2:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[IN_DIM2]]]
//      CHECK:   return %[[C12]], %[[OUT_DIM1]], %[[OUT_DIM2]]

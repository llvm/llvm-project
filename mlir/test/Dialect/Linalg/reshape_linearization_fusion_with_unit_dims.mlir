// RUN: mlir-opt -linalg-fold-reshape-ops-by-linearization=allow-folding-unit-dim-reshapes -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func @do_not_fold1(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?x1xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
  %4 = tensor.expand_shape %3 [[0], [1, 2]] : tensor<?x?xf32> into tensor<?x?x1xf32>
  return %4 : tensor<?x?x1xf32>
}
// CHECK-LABEL: func @do_not_fold1
//       CHECK: %[[VAL:.+]] = linalg.generic
//       CHECK: tensor.expand_shape %[[VAL]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func @do_not_fold2(%arg0 : tensor<?x?x1xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.collapse_shape %arg0 [[0], [1, 2]] : tensor<?x?x1xf32> into tensor<?x?xf32>
  %1 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %3 = linalg.init_tensor [%1, %2] : tensor<?x?xf32>
  %4 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]}
      ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%3 : tensor<?x?xf32>) {
      ^bb0(%arg2 : f32, %arg3 : f32, %arg4 : f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func @do_not_fold2
//       CHECK: %[[VAL:.+]] = tensor.collapse_shape
//       CHECK: linalg.generic
//  CHECK-SAME:   ins(%[[VAL]], %{{.+}} : tensor<?x?xf32>, tensor<?x?xf32>)

// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-reassociative-reshape-folding %s | FileCheck %s

// CHECK-LABEL: func @expand_shape_of_rank_reducing_extract(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?x?xf32>
//   CHECK-DAG:   %[[extract1:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x1x1x5xf32>
//   CHECK-DAG:   %[[extract2:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0, 0] [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x?x?x?xf32> to tensor<?x1x1x5xf32>
//       CHECK:   return %[[extract1]], %[[extract2]]
func.func @expand_shape_of_rank_reducing_extract(
    %t: tensor<?x?x?x?xf32>, %idx: index)
  -> (tensor<?x1x1x5xf32>, tensor<?x1x1x5xf32>)
{
  %0 = tensor.extract_slice %t[0, 0, 0, 0][%idx, 1, 1, 5][1, 1, 1, 1]
      : tensor<?x?x?x?xf32> to tensor<?x1x5xf32>
  %1 = tensor.expand_shape %0 [[0], [1, 2], [3]]
      : tensor<?x1x5xf32> into tensor<?x1x1x5xf32>
  %2 = tensor.expand_shape %0 [[0, 1], [2], [3]]
      : tensor<?x1x5xf32> into tensor<?x1x1x5xf32>
  return %1, %2 : tensor<?x1x1x5xf32>, tensor<?x1x1x5xf32>
}

// -----

// CHECK-LABEL: func @rank_reducing_insert_of_collapse_shape(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x1x1x5xf32>
//       CHECK:   %[[insert:.*]] = tensor.insert_slice %[[t]] into %{{.*}}[0, 0, 0, 0] [%{{.*}}, 1, 1, 5] [1, 1, 1, 1] : tensor<?x1x1x5xf32> into tensor<?x?x?x?xf32>
//       CHECK:   return %[[insert]]
func.func @rank_reducing_insert_of_collapse_shape(
    %t: tensor<?x1x1x5xf32>, %d: tensor<?x?x?x?xf32>, %sz: index)
  -> tensor<?x?x?x?xf32> {
  %0 = tensor.collapse_shape %t [[0, 1], [2], [3]]
      : tensor<?x1x1x5xf32> into tensor<?x1x5xf32>
  %1 = tensor.insert_slice %0 into %d[0, 0, 0, 0][%sz, 1, 1, 5][1, 1, 1, 1]
      : tensor<?x1x5xf32> into tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}

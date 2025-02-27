// RUN: mlir-opt -slice-analysis-test -split-input-file %s | FileCheck %s

func.func @slicing_linalg_op(%arg0 : index, %arg1 : index, %arg2 : index) {
  %a = memref.alloc(%arg0, %arg2) : memref<?x?xf32>
  %b = memref.alloc(%arg2, %arg1) : memref<?x?xf32>
  %c = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %d = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%c : memref<?x?xf32>)
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%d : memref<?x?xf32>)
  memref.dealloc %c : memref<?x?xf32>
  memref.dealloc %b : memref<?x?xf32>
  memref.dealloc %a : memref<?x?xf32>
  memref.dealloc %d : memref<?x?xf32>
  return
}

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__0
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = memref.alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = memref.alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = memref.alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return

// CHECK-LABEL: func @slicing_linalg_op__backward_slice__1
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[A:.+]] = memref.alloc(%[[ARG0]], %[[ARG2]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[B:.+]] = memref.alloc(%[[ARG2]], %[[ARG1]]) : memref<?x?xf32>
//   CHECK-DAG:   %[[C:.+]] = memref.alloc(%[[ARG0]], %[[ARG1]]) : memref<?x?xf32>
//       CHECK:   return

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @slice_use_from_above(%arg0: tensor<5x5xf32>, %arg1: tensor<5x5xf32>) {
  %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.addf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<5x5xf32> into tensor<25xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>) {
  ^bb0(%in: f32, %out: f32):
    %c2 = arith.constant 2 : index
    %extracted = tensor.extract %collapsed[%c2] : tensor<25xf32>
    %2 = arith.addf %extracted, %extracted : f32
    linalg.yield %2 : f32
  } -> tensor<5x5xf32>
  return
}

// CHECK-LABEL: func @slice_use_from_above__backward_slice__0
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor 
//       CHECK:   %[[A:.+]] = linalg.generic {{.*}} ins(%[[ARG0]]
//       CHECK:   %[[B:.+]] = tensor.collapse_shape %[[A]]
//       CHECK:   return

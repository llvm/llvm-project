//RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-swap-extract-slice-with-fill-pattern %s | FileCheck %s

// CHECK-LABEL: func.func @swap_fill_insert_slice
//  CHECK-SAME: (%[[INIT:.+]]: tensor<?x?x?xf32>, %[[OFFSET0:.+]]: index, %[[SIZE1:.+]]: index)
//       CHECK:   %[[F0:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EXT:.+]] = tensor.extract_slice %[[INIT]][%[[OFFSET0]], 8, 4] [1, %[[SIZE1]], 6] [1, 3, 1]
//       CHECK:   %[[FILL:.+]] = linalg.fill ins(%[[F0]] : f32) outs(%[[EXT]] : tensor<?x6xf32>) -> tensor<?x6xf32>
//       CHECK:   return %[[FILL]]
func.func @swap_fill_insert_slice(%init : tensor<?x?x?xf32>, %offset0: index, %size1: index) -> tensor<?x6xf32> {
  %f0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%f0 : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = tensor.extract_slice %0[%offset0, 8, 4] [1, %size1, 6] [1, 3, 1]
    : tensor<?x?x?xf32> to tensor<?x6xf32>
  return %1: tensor<?x6xf32>
}

// -----

// CHECK-LABEL: func.func @dont_swap_fill_insert_slice_multi_user
//       CHECK:   linalg.fill
//       CHECK:   tensor.extract_slice
func.func @dont_swap_fill_insert_slice_multi_user(%init : tensor<?x?x?xf32>, %offset0: index, %size1: index) -> (tensor<?x?x?xf32>, tensor<2x?x6xf32>) {
  %f0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%f0 : f32) outs(%init : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = tensor.extract_slice %0[%offset0, 8, 4] [2, %size1, 6] [1, 3, 1]
    : tensor<?x?x?xf32> to tensor<2x?x6xf32>
  return %0, %1: tensor<?x?x?xf32>, tensor<2x?x6xf32>
}

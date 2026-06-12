//RUN: mlir-opt -test-linalg-rank-reduce-contraction-ops --canonicalize -split-input-file %s | FileCheck %s

func.func @singleton_batch_matmul_tensor(%arg0 : tensor<1x128x512xf32>, %arg1 : tensor<1x512x256xf32>, %arg2: tensor<1x128x256xf32>) -> tensor<1x128x256xf32> {
  // CHECK-LABEL: @singleton_batch_matmul_tensor
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<1x128x512xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<1x512x256xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<1x128x256xf32>
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = tensor.collapse_shape %[[LHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_RHS:.*]] = tensor.collapse_shape %[[RHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[MATMUL:.+]] = linalg.matmul ins(%[[COLLAPSED_LHS]], %[[COLLAPSED_RHS]] : tensor<128x512xf32>, tensor<512x256xf32>) outs(%[[COLLAPSED_INIT]] : tensor<128x256xf32>)
  //  CHECK-NEXT:   %[[RES:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]] output_shape [1, 128, 256]
  //  CHECK-NEXT:   return %[[RES]]
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x128x512xf32>, tensor<1x512x256xf32>)
      outs(%arg2 : tensor<1x128x256xf32>) -> tensor<1x128x256xf32>
  return %1 : tensor<1x128x256xf32>
}

// -----

func.func @singleton_batch_matmul_memref(%arg0 : memref<1x?x?xf32>, %arg1 : memref<1x?x?xf32>, %arg2: memref<1x?x?xf32>) {
  // CHECK-LABEL: @singleton_batch_matmul_memref
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: memref<1x?x?xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: memref<1x?x?xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: memref<1x?x?xf32>
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = memref.collapse_shape %[[LHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_RHS:.*]] = memref.collapse_shape %[[RHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = memref.collapse_shape %[[INIT]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:    linalg.matmul ins(%[[COLLAPSED_LHS]], %[[COLLAPSED_RHS]] : memref<?x?xf32>, memref<?x?xf32>) outs(%[[COLLAPSED_INIT]] : memref<?x?xf32>)
  //  CHECK-NEXT:   return
  linalg.batch_matmul ins(%arg0, %arg1 : memref<1x?x?xf32>, memref<1x?x?xf32>)
      outs(%arg2 : memref<1x?x?xf32>)
  return
}

// -----

func.func @negative_singleton_batch_matmul_to_matmul_memref(%arg0 : memref<1x?x?xf32>, %arg1 : memref<1x?x?xf32>, %arg2: memref<1x?x?xf32>) {
  // CHECK-LABEL: @negative_singleton_batch_matmul_to_matmul_memref
  // CHECK-NOT:   collapse_shape
  // CHECK-NOT:   linalg.matmul
  // CHECK-NOT:   expand_shape
  linalg.batch_matmul indexing_maps = [
                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                          affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
                          ]
      ins(%arg0, %arg1 : memref<1x?x?xf32>, memref<1x?x?xf32>)
      outs(%arg2 : memref<1x?x?xf32>)
  return
}

// -----

func.func @singleton_batch_matvec(%arg0 : tensor<1x128x512xf32>, %arg1 : tensor<1x512xf32>, %arg2: tensor<1x128xf32>) -> tensor<1x128xf32> {
  // CHECK-LABEL: @singleton_batch_matvec
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<1x128x512xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<1x512xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<1x128xf32>
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = tensor.collapse_shape %[[LHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_RHS:.*]] = tensor.collapse_shape %[[RHS]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[MATMUL:.+]] = linalg.matvec
  //  CHECK-SAME:   ins(%[[COLLAPSED_LHS]], %[[COLLAPSED_RHS]] : tensor<128x512xf32>, tensor<512xf32>) outs(%[[COLLAPSED_INIT]] : tensor<128xf32>)
  //  CHECK-NEXT:   %[[RES:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1]] output_shape [1, 128]
  //  CHECK-NEXT:   return %[[RES]]
  %1 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<1x128x512xf32>, tensor<1x512xf32>)
      outs(%arg2 : tensor<1x128xf32>) -> tensor<1x128xf32>
  return %1 : tensor<1x128xf32>
}

// -----

func.func @singleton_batch_vecmat(%arg0 : tensor<1x?xf32>, %arg1 : tensor<1x?x?xf32>, %arg2: tensor<1x?xf32>) -> tensor<1x?xf32> {
  // CHECK-LABEL: @singleton_batch_vecmat
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<1x?xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<1x?x?xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<1x?xf32>
  //  CHECK-DAG:    %[[C0:.*]] = arith.constant 0
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = tensor.collapse_shape %[[LHS]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[COLLAPSED_RHS:.*]] = tensor.collapse_shape %[[RHS]] {{\[}}[0, 1], [2]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[MATMUL:.+]] = linalg.vecmat 
  //  CHECK-SAME:   ins(%[[COLLAPSED_LHS]], %[[COLLAPSED_RHS]] : tensor<?xf32>, tensor<?x?xf32>) outs(%[[COLLAPSED_INIT]] : tensor<?xf32>)
  //  CHECK-NEXT:   %[[DIM1:.*]] = tensor.dim %[[COLLAPSED_INIT]], %[[C0]]
  //  CHECK-NEXT:   %[[RES:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1]] output_shape [1, %[[DIM1]]]
  //  CHECK-NEXT:   return %[[RES]]
  %1 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<1x?xf32>, tensor<1x?x?xf32>)
      outs(%arg2 : tensor<1x?xf32>) -> tensor<1x?xf32>
  return %1 : tensor<1x?xf32>
}

// -----

func.func @matmul_to_matvec_tensor(%arg0: tensor<?x?xf32>, %arg1: tensor<?x1xf32>, %arg2: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK-LABEL: @matmul_to_matvec_tensor
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x1xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<?x1xf32>
  //  CHECK-DAG:    %[[C0:.*]] = arith.constant 0
  //  CHECK-NEXT:   %[[COLLAPSED_RHS:.*]] = tensor.collapse_shape %[[RHS]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[MATMUL:.+]] = linalg.matvec 
  //  CHECK-SAME:   ins(%[[LHS]], %[[COLLAPSED_RHS]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[COLLAPSED_INIT]] : tensor<?xf32>)
  //  CHECK-NEXT:   %[[DIM0:.*]] = tensor.dim %[[COLLAPSED_INIT]], %[[C0]]
  //  CHECK-NEXT:   %[[RES:.*]] = tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1]] output_shape [%[[DIM0]], 1]
  //  CHECK-NEXT:   return %[[RES]]
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x1xf32>) outs(%arg2: tensor<?x1xf32>) -> tensor<?x1xf32>
    return %0 : tensor<?x1xf32>
}

// -----

func.func @matmul_to_matvec(%arg0: memref<?x?xf32>, %arg1: memref<?x1xf32>, %arg2: memref<?x1xf32>) {
  // CHECK-LABEL: @matmul_to_matvec
  // CHECK: linalg.matvec
    linalg.matmul ins(%arg0, %arg1: memref<?x?xf32>, memref<?x1xf32>) outs(%arg2: memref<?x1xf32>)
    return
}

// -----

func.func @negative_matmul_to_matvec(%arg0: memref<?xf32>, %arg1: memref<?x1xf32>, %arg2: memref<?x1xf32>) {
  // CHECK-LABEL: @negative_matmul_to_matvec
  // CHECK-NOT: linalg.matvec
    linalg.matmul indexing_maps = [
                          affine_map<(d0, d1, d2) -> (d2)>,
                          affine_map<(d0, d1, d2) -> (d2, d1)>,
                          affine_map<(d0, d1, d2) -> (d0, d1)>
                          ]
                          ins(%arg0, %arg1: memref<?xf32>, memref<?x1xf32>) outs(%arg2: memref<?x1xf32>)
    return
}

// -----

func.func @matmul_to_vecmat_tensor(%arg0: tensor<1x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<1x?xf32>) -> tensor<1x?xf32> {
  // CHECK-LABEL: @matmul_to_vecmat
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<1x?xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: tensor<1x?xf32>
  //  CHECK-DAG:    %[[C0:.*]] = arith.constant 0
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = tensor.collapse_shape %[[LHS]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = tensor.collapse_shape %[[INIT]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[RESULT:.*]] = linalg.vecmat 
  //  CHECK-SAME:   ins(%[[COLLAPSED_LHS]], %[[RHS]] : tensor<?xf32>, tensor<?x?xf32>) outs(%[[COLLAPSED_INIT]] : tensor<?xf32>)
  //  CHECK-NEXT:   %[[DIM1:.*]] = tensor.dim %[[COLLAPSED_INIT]], %[[C0]]
  //  CHECK-NEXT:   %[[RES:.*]] = tensor.expand_shape %[[RESULT]] {{\[}}[0, 1]] output_shape [1, %[[DIM1]]]
  //  CHECK-NEXT:   return %[[RES]]
    %0 = linalg.matmul ins(%arg0, %arg1: tensor<1x?xf32>, tensor<?x?xf32>) outs(%arg2: tensor<1x?xf32>) -> tensor<1x?xf32>
    return %0 : tensor<1x?xf32>
}

// -----

func.func @batch_matmul_to_vecmat(%arg0: memref<1x1x?xf32>, %arg1: memref<1x?x?xf32>, %arg2: memref<1x1x?xf32>) {
  // CHECK-LABEL: @batch_matmul_to_vecmat
  // CHECK: linalg.vecmat
    linalg.batch_matmul ins(%arg0, %arg1: memref<1x1x?xf32>, memref<1x?x?xf32>) outs(%arg2: memref<1x1x?xf32>)
    return
}

// -----

func.func @matvec_to_dot(%arg0: memref<1x?xf32>, %arg1: memref<?xf32>, %arg2: memref<1xf32>) {
  // CHECK-LABEL: @matvec_to_dot
  //  CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: memref<1x?xf32>
  //  CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: memref<?xf32>
  //  CHECK-SAME:     %[[INIT:[a-zA-Z0-9]+]]: memref<1xf32>
  //  CHECK-NEXT:   %[[COLLAPSED_LHS:.*]] = memref.collapse_shape %[[LHS]] {{\[}}[0, 1]]
  //  CHECK-NEXT:   %[[COLLAPSED_INIT:.*]] = memref.collapse_shape %[[INIT]] []
  //  CHECK-NEXT:   linalg.dot ins(%[[COLLAPSED_LHS]], %[[RHS]] : memref<?xf32>, memref<?xf32>) outs(%[[COLLAPSED_INIT]] : memref<f32>)
    linalg.matvec ins(%arg0, %arg1: memref<1x?xf32>, memref<?xf32>) outs(%arg2: memref<1xf32>)
    return
}

// -----

func.func @vecmat_to_dot(%arg0: memref<?xf32>, %arg1: memref<?x1xf32>, %arg2: memref<1xf32>) {
  // CHECK-LABEL: @vecmat_to_dot
  // CHECK: linalg.dot
    linalg.vecmat ins(%arg0, %arg1: memref<?xf32>, memref<?x1xf32>) outs(%arg2: memref<1xf32>)
    return
}

// -----

func.func @matvec_to_dot_tensor(%arg0: tensor<1x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK-LABEL: @matvec_to_dot_tensor
  // CHECK: linalg.dot
    %0 = linalg.matvec ins(%arg0, %arg1: tensor<1x?xf32>, tensor<?xf32>) outs(%arg2: tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
}

// -----

func.func @nonsingleton_batch_matmul(%arg0 : tensor<2x?x?xf32>, %arg1 : tensor<2x?x?xf32>, %arg2: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> {
  // CHECK-LABEL: @nonsingleton_batch_matmul
  // CHECK-NOT:   collapse_shape
  // CHECK:       linalg.batch_matmul
  // CHECK-NOT:   expand_shape
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<2x?x?xf32>, tensor<2x?x?xf32>)
      outs(%arg2 : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  return %1 : tensor<2x?x?xf32>
}

// -----

func.func @nonsingleton_batch_matmul_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-LABEL: @nonsingleton_batch_matmul_dynamic
  // CHECK-NOT:   collapse_shape
  // CHECK:       linalg.batch_matmul
  // CHECK-NOT:   expand_shape
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

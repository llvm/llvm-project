// RUN: mlir-opt -fold-memref-alias-ops -split-input-file %s | FileCheck %s

func.func @fold_gpu_subgroup_mma_load_matrix_1d(%src: memref<?xvector<4xf32>>, %offset: index, %i: index) -> !gpu.mma_matrix<16x16xf16, "COp"> {
  %subview = memref.subview %src[%offset] [81920] [1] : memref<?xvector<4xf32>> to memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  %matrix = gpu.subgroup_mma_load_matrix %subview[%i] {leadDimension = 160 : index} : memref<81920xvector<4xf32>, strided<[1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
  return %matrix: !gpu.mma_matrix<16x16xf16, "COp">
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func.func @fold_gpu_subgroup_mma_load_matrix_1d
// CHECK-SAME: (%[[SRC:.+]]: memref<?xvector<4xf32>>, %[[OFFSET:.+]]: index, %[[I:.+]]: index)
//      CHECK:   %[[APPLY:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]], %[[I]]]
//      CHECK:   %[[LOAD:.+]] = gpu.subgroup_mma_load_matrix %[[SRC]][%[[APPLY]]] {leadDimension = 160 : index} : memref<?xvector<4xf32>> -> !gpu.mma_matrix<16x16xf16, "COp">
//      CHECK:   return %[[LOAD]]

// -----

func.func @fold_gpu_subgroup_mma_store_matrix_1d(%dst: memref<?xvector<4xf32>>, %offset: index, %i: index, %matrix: !gpu.mma_matrix<16x16xf16, "COp">) {
  %subview = memref.subview %dst[%offset] [81920] [1] : memref<?xvector<4xf32>> to memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  gpu.subgroup_mma_store_matrix %matrix, %subview[%i] {leadDimension = 160 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<81920xvector<4xf32>, strided<[1], offset: ?>>
  return
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func.func @fold_gpu_subgroup_mma_store_matrix_1d
// CHECK-SAME: (%[[DST:.+]]: memref<?xvector<4xf32>>, %[[OFFSET:.+]]: index, %[[I0:.+]]: index, %[[VAL:.+]]: !gpu.mma_matrix<16x16xf16, "COp">)
//      CHECK:   %[[APPLY:.+]] = affine.apply #[[MAP]]()[%[[OFFSET]], %[[I0]]]
//      CHECK:   gpu.subgroup_mma_store_matrix %[[VAL]], %[[DST]][%[[APPLY]]] {leadDimension = 160 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<?xvector<4xf32>>

// -----

// CHECK-LABEL: func.func @fold_gpu_subgroup_mma_load_matrix_2d
//  CHECK-SAME: %[[SRC:.+]]: memref<128x128xf32>
func.func @fold_gpu_subgroup_mma_load_matrix_2d(%arg0 : memref<128x128xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> !gpu.mma_matrix<16x16xf16, "COp"> {
  %subview = memref.subview %arg0[%arg1, %arg2][64, 32][2, 1] : memref<128x128xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
  // CHECK: gpu.subgroup_mma_load_matrix %[[SRC]][{{.+}}] {leadDimension = 32 : index} : memref<128x128xf32> -> !gpu.mma_matrix<16x16xf16, "COp">
  %matrix = gpu.subgroup_mma_load_matrix %subview[%arg3, %arg4] {leadDimension = 32 : index} : memref<64x32xf32, strided<[256, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf16, "COp">
  return %matrix : !gpu.mma_matrix<16x16xf16, "COp">
}

// -----

// CHECK-LABEL: func.func @fold_gpu_subgroup_mma_load_matrix_2d
//  CHECK-SAME: %[[DST:.+]]: memref<128x128xf32>
func.func @fold_gpu_subgroup_mma_load_matrix_2d(%arg0 : memref<128x128xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %matrix: !gpu.mma_matrix<16x16xf16, "COp">) {
  %subview = memref.subview %arg0[%arg1, %arg2][64, 32][2, 1] : memref<128x128xf32> to memref<64x32xf32, strided<[256, 1], offset: ?>>
  // CHECK: gpu.subgroup_mma_store_matrix %{{.+}}, %[[DST]][{{.+}}] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<128x128xf32>
  gpu.subgroup_mma_store_matrix %matrix, %subview[%arg3, %arg4] {leadDimension = 32 : index} :  !gpu.mma_matrix<16x16xf16, "COp">, memref<64x32xf32, strided<[256, 1], offset: ?>>
  return
}

// -----

func.func @fold_gpu_subgroup_mma_load_matrix_expand_shape(%src: memref<4096xf32>, %i: index, %j: index) -> !gpu.mma_matrix<16x16xf16, "COp"> {
  %expand = memref.expand_shape %src [[0, 1]] output_shape [64, 64] : memref<4096xf32> into memref<64x64xf32>
  %matrix = gpu.subgroup_mma_load_matrix %expand[%i, %j] {leadDimension = 64 : index} : memref<64x64xf32> -> !gpu.mma_matrix<16x16xf16, "COp">
  return %matrix: !gpu.mma_matrix<16x16xf16, "COp">
}

//      CHECK: func.func @fold_gpu_subgroup_mma_load_matrix_expand_shape
// CHECK-SAME: (%[[SRC:.+]]: memref<4096xf32>, %[[I:.+]]: index, %[[J:.+]]: index)
//      CHECK:   %[[LIN:.+]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (64, 64)
//      CHECK:   %[[LOAD:.+]] = gpu.subgroup_mma_load_matrix %[[SRC]][%[[LIN]]] {leadDimension = 64 : index}
//      CHECK:   return %[[LOAD]]

// -----

func.func @fold_gpu_subgroup_mma_store_matrix_expand_shape(%dst: memref<4096xf32>, %i: index, %j: index, %matrix: !gpu.mma_matrix<16x16xf16, "COp">) {
  %expand = memref.expand_shape %dst [[0, 1]] output_shape [64, 64] : memref<4096xf32> into memref<64x64xf32>
  gpu.subgroup_mma_store_matrix %matrix, %expand[%i, %j] {leadDimension = 64 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<64x64xf32>
  return
}

//      CHECK: func.func @fold_gpu_subgroup_mma_store_matrix_expand_shape
// CHECK-SAME: (%[[DST:.+]]: memref<4096xf32>, %[[I:.+]]: index, %[[J:.+]]: index, %[[MATRIX:.+]]: !gpu.mma_matrix<16x16xf16, "COp">)
//      CHECK:   %[[LIN:.+]] = affine.linearize_index disjoint [%[[I]], %[[J]]] by (64, 64)
//      CHECK:   gpu.subgroup_mma_store_matrix %[[MATRIX]], %[[DST]][%[[LIN]]] {leadDimension = 64 : index}
//      CHECK:   return


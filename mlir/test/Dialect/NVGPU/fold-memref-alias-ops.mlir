// RUN: mlir-opt -fold-memref-alias-ops -split-input-file %s | FileCheck %s

func.func @fold_nvgpu_device_async_copy_zero_sub_idx(%gmem_memref_3d : memref<2x128x768xf16>, %idx_1 : index, %idx_2 : index, %idx_3 : index) {
  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%idx_1, %idx_2, %idx_3] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%c0, %c0], %smem_memref_4d[%c0, %c0, %c0, %c0], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  return
}

// CHECK-LABEL: func.func @fold_nvgpu_device_async_copy_zero_sub_idx
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[IDX_1:.+]]: index, %[[IDX_2:.+]]: index, %[[IDX_3:.+]]: index)
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[SMEM_MEMREF_4d:.+]] = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
//       CHECK: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[IDX_1]], %[[IDX_2]], %[[IDX_3]]], %[[SMEM_MEMREF_4d]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----


func.func @fold_src_nvgpu_device_async_copy(%gmem_memref_3d : memref<2x128x768xf16>, %src_idx_0 : index, %src_idx_1 : index, %src_idx_2 : index, %src_sub_idx_0 : index, %src_sub_idx_1 : index) {
  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%src_idx_0, %src_idx_1, %src_idx_2] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%src_sub_idx_0, %src_sub_idx_1], %smem_memref_4d[%c0, %c0, %c0, %c0], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  return
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func.func @fold_src_nvgpu_device_async_copy
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[SRC_IDX_0:.+]]: index, %[[SRC_IDX_1:.+]]: index, %[[SRC_IDX_2:.+]]: index, %[[SRC_SUB_IDX_0:.+]]: index, %[[SRC_SUB_IDX_1:.+]]: index)
//   CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_0:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_0]], %[[SRC_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_2]], %[[SRC_SUB_IDX_1]]]
//   CHECK-DAG: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[RESOLVED_SRC_IDX_0]], %[[SRC_IDX_1]], %[[RESOLVED_SRC_IDX_1]]], %[[SMEM_MEMREF_4d]][%[[c0]], %[[c0]], %[[c0]], %[[c0]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----


func.func @fold_src_fold_dest_nvgpu_device_async_copy(%gmem_memref_3d : memref<2x128x768xf16>, %src_idx_0 : index, %src_idx_1 : index, %src_idx_2 : index, %src_sub_idx_0 : index, %src_sub_idx_1 : index, %dest_idx_0 : index, %dest_idx_1 : index, %dest_idx_2 : index, %dest_idx_3 : index, %dest_sub_idx_0 : index, %dest_sub_idx_1 : index) {
  %c0 = arith.constant 0 : index
  %smem_memref_4d = memref.alloc() : memref<5x1x64x64xf16, #gpu.address_space<workgroup>>
  %gmem_memref_subview_2d = memref.subview %gmem_memref_3d[%src_idx_0, %src_idx_1, %src_idx_2] [1, 1, 8] [1, 1, 1] : memref<2x128x768xf16> to memref<1x8xf16, strided<[98304, 1], offset: ?>>
  %smem_memref_2d = memref.subview %smem_memref_4d[%dest_idx_0, %dest_idx_1, %dest_idx_2, %dest_idx_3] [1, 1, 1, 8] [1, 1, 1, 1] : memref<5x1x64x64xf16, #gpu.address_space<workgroup>> to memref<1x8xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<workgroup>>
  %async_token = nvgpu.device_async_copy %gmem_memref_subview_2d[%src_sub_idx_0, %src_sub_idx_1], %smem_memref_2d[%dest_sub_idx_0, %dest_sub_idx_1], 8 {bypassL1} : memref<1x8xf16, strided<[98304, 1], offset: ?>> to memref<1x8xf16, strided<[4096, 1], offset: ?>, #gpu.address_space<workgroup>>
  return
}

//   CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func.func @fold_src_fold_dest_nvgpu_device_async_copy
//  CHECK-SAME: (%[[GMEM_MEMREF_3d:.+]]: memref<2x128x768xf16>, %[[SRC_IDX_0:.+]]: index, %[[SRC_IDX_1:.+]]: index, %[[SRC_IDX_2:.+]]: index, %[[SRC_SUB_IDX_0:.+]]: index, %[[SRC_SUB_IDX_1:.+]]: index, %[[DEST_IDX_0:.+]]: index, %[[DEST_IDX_1:.+]]: index, %[[DEST_IDX_2:.+]]: index, %[[DEST_IDX_3:.+]]: index, %[[DEST_SUB_IDX_0:.+]]: index, %[[DEST_SUB_IDX_1:.+]]: index)
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_0:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_0]], %[[SRC_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_SRC_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[SRC_IDX_2]], %[[SRC_SUB_IDX_1]]]
//   CHECK-DAG: %[[RESOLVED_DST_IDX_1:.+]] = affine.apply #[[MAP]]()[%[[DEST_IDX_1]], %[[DEST_SUB_IDX_0]]]
//   CHECK-DAG: %[[RESOLVED_DST_IDX_3:.+]] = affine.apply #[[MAP]]()[%[[DEST_IDX_3]], %[[DEST_SUB_IDX_1]]]
//   CHECK-DAG: nvgpu.device_async_copy %[[GMEM_MEMREF_3d]][%[[RESOLVED_SRC_IDX_0]], %[[SRC_IDX_1]], %[[RESOLVED_SRC_IDX_1]]], %[[SMEM_MEMREF_4d]][%[[DEST_IDX_0]], %[[RESOLVED_DST_IDX_1]], %[[DEST_IDX_2]], %[[RESOLVED_DST_IDX_3]]], 8 {bypassL1} : memref<2x128x768xf16> to memref<5x1x64x64xf16, #gpu.address_space<workgroup>>

// -----

#map = affine_map<()[s0] -> (-s0 + 4)>
#map1 = affine_map<()[s0] -> (-s0 + 32)>

func.func @test_ldmatrix(%arg0: memref<4x32x32xf16, 3>, %arg1: index, %arg2: index, %arg3: index) -> vector<4x2xf16> {
  %c0 = arith.constant 0 : index
  %0 = affine.apply #map()[%arg1]
  %1 = affine.apply #map1()[%arg2]
  %2 = affine.apply #map1()[%arg3]
  %subview = memref.subview %arg0[%arg1, %arg2, %arg3] [%0, %1, %2] [1, 1, 1] : memref<4x32x32xf16, 3> to memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3>
  %3 = nvgpu.ldmatrix %subview[%c0, %c0, %c0] {numTiles = 4 : i32, transpose = false} : memref<?x?x?xf16, strided<[1024, 32, 1], offset: ?>, 3> -> vector<4x2xf16>
  return %3 : vector<4x2xf16>
}

//      CHECK: func @test_ldmatrix
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x32x32xf16, 3>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   nvgpu.ldmatrix %[[ARG0]][%[[ARG1]], %[[ARG2]], %[[ARG3]]] {numTiles = 4 : i32, transpose = false} : memref<4x32x32xf16, 3> -> vector<4x2xf16>

// -----

func.func @ldmatrix_expand(%arg0: memref<4096xf16, 3>, %arg1: index, %arg2: index, %arg3: index) -> vector<4x2xf16> {
  %exp = memref.expand_shape %arg0 [[0, 1, 2]] output_shape [4, 32, 32] : memref<4096xf16, 3> into memref<4x32x32xf16, 3>
  %3 = nvgpu.ldmatrix %exp[%arg1, %arg2, %arg3] {numTiles = 4 : i32, transpose = false} : memref<4x32x32xf16, 3> -> vector<4x2xf16>
  return %3 : vector<4x2xf16>
}

//      CHECK: func @ldmatrix_expand
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4096xf16, 3>
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[LIN:[a-zA-Z0-9_]+]] = affine.linearize_index disjoint [%[[ARG1]], %[[ARG2]], %[[ARG3]]] by (4, 32, 32)
//      CHECK:   nvgpu.ldmatrix %[[ARG0]][%[[LIN]]] {numTiles = 4 : i32, transpose = false} : memref<4096xf16, 3> -> vector<4x2xf16>

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 allow-padding=1" \
// RUN: -canonicalize | FileCheck %s

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 allow-padding=0" \
// RUN: -canonicalize | FileCheck %s --check-prefix=NOPAD

// RUN: mlir-opt %s -linalg-block-pack-matmul="block-factors=32,16,64 allow-padding=1 mnk-padded-multiples=256,512,384" \
// RUN: -canonicalize | FileCheck %s --check-prefix=PAD-MULT

func.func @block_matmul_padding(
    %A: tensor<123x125xf32>, %B: tensor<125x124xf32>, %C: tensor<123x124xf32>) -> tensor<123x124xf32> {
  %0 = linalg.matmul  ins(%A, %B : tensor<123x125xf32>, tensor<125x124xf32>)
                      outs(%C : tensor<123x124xf32>) -> tensor<123x124xf32>
  return %0 : tensor<123x124xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-LABEL: func @block_matmul_padding(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<123x125xf32>, %[[B:[0-9a-z]+]]: tensor<125x124xf32>, %[[C:[0-9a-z]+]]: tensor<123x124xf32>
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<4x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = tensor.pack %[[A]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<123x125xf32> -> tensor<4x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<8x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = tensor.pack %[[B]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<125x124xf32> -> tensor<8x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = tensor.pack %[[C]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<123x124xf32> -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<4x2x32x64xf32>, tensor<8x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<123x124xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<123x124xf32>

// NOPAD-LABEL: func @block_matmul_padding(
// NOPAD-SAME:    %[[A:[0-9a-z]+]]: tensor<123x125xf32>, %[[B:[0-9a-z]+]]: tensor<125x124xf32>, %[[C:[0-9a-z]+]]: tensor<123x124xf32>
// NOPAD-NOT: tensor.pack
// NOPAD: linalg.matmul ins(%[[A]], %[[B]] : tensor<123x125xf32>, tensor<125x124xf32>)
// NOPAD-SAME: outs(%[[C]] : tensor<123x124xf32>) -> tensor<123x124xf32>
// NOPAD-NOT: tensor.unpack

// PAD-MULT-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// PAD-MULT-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// PAD-MULT-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// PAD-MULT-LABEL: func @block_matmul_padding(
// PAD-MULT-SAME:    %[[A:[0-9a-z]+]]: tensor<123x125xf32>, %[[B:[0-9a-z]+]]: tensor<125x124xf32>, %[[C:[0-9a-z]+]]: tensor<123x124xf32>
// PAD-MULT-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// PAD-MULT: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<1x1x256x384xf32>
// PAD-MULT: %[[A_PACKED:.+]] = tensor.pack %[[A]]
// PAD-MULT-SAME:  padding_value(%[[ZERO]] : f32)
// PAD-MULT-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [256, 384]
// PAD-MULT-SAME:  into %[[PACK_DST_0]] : tensor<123x125xf32> -> tensor<1x1x256x384xf32>
// PAD-MULT: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<1x1x512x384xf32>
// PAD-MULT: %[[B_PACKED:.+]] = tensor.pack %[[B]]
// PAD-MULT-SAME:  padding_value(%[[ZERO]] : f32)
// PAD-MULT-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [512, 384]
// PAD-MULT-SAME:  into %[[PACK_DST_1]] : tensor<125x124xf32> -> tensor<1x1x512x384xf32>
// PAD-MULT: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<1x1x256x512xf32>
// PAD-MULT: %[[C_PACKED:.+]] = tensor.pack %[[C]]
// PAD-MULT-SAME:  padding_value(%[[ZERO]] : f32)
// PAD-MULT-SAME:  inner_dims_pos = [0, 1] inner_tiles = [256, 512]
// PAD-MULT-SAME:  into %[[PACK_DST_2]] : tensor<123x124xf32> -> tensor<1x1x256x512xf32>
// PAD-MULT: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// PAD-MULT-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// PAD-MULT-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// PAD-MULT-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<1x1x256x384xf32>, tensor<1x1x512x384xf32>) outs(%[[C_PACKED]] : tensor<1x1x256x512xf32>)
// PAD-MULT: %[[RES_UNPACKED:.+]] = tensor.unpack %[[GEMM_RES_PACKED]]
// PAD-MULT-SAME:  inner_dims_pos = [0, 1] inner_tiles = [256, 512]
// PAD-MULT-SAME:  into %[[C]] : tensor<1x1x256x512xf32> -> tensor<123x124xf32>
// PAD-MULT: return %[[RES_UNPACKED]] : tensor<123x124xf32>

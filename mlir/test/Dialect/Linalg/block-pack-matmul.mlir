// RUN: mlir-opt %s -linalg-block-pack-matmul=block-factors=32,16,64 -canonicalize -split-input-file | FileCheck %s

func.func @block_matmul(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_matmul(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<4x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<128x128xf32> -> tensor<4x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<8x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<128x128xf32> -> tensor<8x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<128x128xf32> -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<4x2x32x64xf32>, tensor<8x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_dynamic(
    %A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[$MAP_M:.+]] = affine_map<()[s0] -> (s0 ceildiv 32)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<()[s0] -> (s0 ceildiv 64)>
// CHECK-DAG: #[[$MAP_N:.+]] = affine_map<()[s0] -> (s0 ceildiv 16)>
// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_matmul_dynamic(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<?x?xf32>, %[[B:[0-9a-z]+]]: tensor<?x?xf32>, %[[C:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[A_M:.+]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG: %[[A_K:.+]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG: %[[A_OUTER_TILE_M:.+]] = affine.apply #[[$MAP_M]]()[%[[A_M]]]
// CHECK-DAG: %[[A_OUTER_TILE_K:.+]] = affine.apply #[[$MAP_K]]()[%[[A_K]]]
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty(%[[A_OUTER_TILE_M]], %[[A_OUTER_TILE_K]]) : tensor<?x?x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<?x?xf32> -> tensor<?x?x32x64xf32>
// CHECK-DAG: %[[B_K:.+]] = tensor.dim %[[B]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG: %[[B_N:.+]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG: %[[B_OUTER_TILE_K:.+]] = affine.apply #[[$MAP_K]]()[%[[B_K]]]
// CHECK-DAG: %[[B_OUTER_TILE_N:.+]] = affine.apply #[[$MAP_N]]()[%[[B_N]]]
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty(%[[B_OUTER_TILE_N]], %[[B_OUTER_TILE_K]]) : tensor<?x?x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<?x?xf32> -> tensor<?x?x16x64xf32>
// CHECK-DAG: %[[C_M:.+]] = tensor.dim %[[C]], %[[C0]] : tensor<?x?xf32>
// CHECK-DAG: %[[C_N:.+]] = tensor.dim %[[C]], %[[C1]] : tensor<?x?xf32>
// CHECK-DAG: %[[C_OUTER_TILE_M:.+]] = affine.apply #[[$MAP_M]]()[%[[C_M]]]
// CHECK-DAG: %[[C_OUTER_TILE_N:.+]] = affine.apply #[[$MAP_N]]()[%[[C_N]]]
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty(%[[C_OUTER_TILE_M]], %[[C_OUTER_TILE_N]]) : tensor<?x?x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  padding_value(%[[ZERO]] : f32)
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<?x?xf32> -> tensor<?x?x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<?x?x32x64xf32>, tensor<?x?x16x64xf32>) outs(%[[C_PACKED]] : tensor<?x?x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<?x?x32x16xf32> -> tensor<?x?xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<?x?xf32>

// -----

func.func @block_matmul_with_constant(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst_acc = arith.constant dense<0.0> : tensor<128x128xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%cst_acc : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_constant(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[CST_ACC_PACKED:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8x32x16xf32>
// CHECK-DAG: %[[RES_DST:.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x16x64xf32>) outs(%[[CST_ACC_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[RES_DST]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_producer(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %acc = linalg.fill ins(%cst : f32) outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%acc : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_producer(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL_DST_PACKED:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[ACC_PACKED:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[FILL_DST_PACKED]] : tensor<4x8x32x16xf32>) -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x16x64xf32>) outs(%[[ACC_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_consumer(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>, %D: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%C : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = linalg.add ins(%1, %D : tensor<128x128xf32>, tensor<128x128xf32>)
                  outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_consumer(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>, %[[D:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[RES_DST:.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  outs({{.*}} : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: %[[ADD_RES:.+]] = linalg.add
// CHECK-SAME:  ins(%[[RES_UNPACKED]], %[[D]] : tensor<128x128xf32>, tensor<128x128xf32>) outs(%[[RES_DST]] : tensor<128x128xf32>)
// CHECK: return %[[ADD_RES]] : tensor<128x128xf32>

// -----

func.func @block_batch_matmul(
    %A: tensor<512x64x128xf32>, %B: tensor<512x128x64xf32>, %C: tensor<512x64x64xf32>) -> tensor<512x64x64xf32> {
  %0 = linalg.batch_matmul ins(%A, %B : tensor<512x64x128xf32>, tensor<512x128x64xf32>)
                           outs(%C : tensor<512x64x64xf32>) -> tensor<512x64x64xf32>
  return %0 : tensor<512x64x64xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d5, d6)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>

// CHECK-LABEL: func @block_batch_matmul(
// CHECK-SAME:   %[[A:.+]]: tensor<512x64x128xf32>, %[[B:.+]]: tensor<512x128x64xf32>, %[[C:.+]]: tensor<512x64x64xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<512x2x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<512x64x128xf32> -> tensor<512x2x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<512x4x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [0, 2, 1] inner_dims_pos = [2, 1] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<512x128x64xf32> -> tensor<512x4x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<512x2x4x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<512x64x64xf32> -> tensor<512x2x4x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<512x2x2x32x64xf32>, tensor<512x4x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<512x2x4x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<512x2x4x32x16xf32> -> tensor<512x64x64xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<512x64x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @block_generic_matmul(
    %A: tensor<128x128xf32>, %B: tensor<128x128xf32>, %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%C : tensor<128x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_generic_matmul(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x128xf32>, %[[B:[0-9a-z]+]]: tensor<128x128xf32>, %[[C:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<4x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<128x128xf32> -> tensor<4x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<8x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<128x128xf32> -> tensor<8x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<128x128xf32> -> tensor<4x8x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<4x2x32x64xf32>, tensor<8x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<4x8x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<128x128xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @block_generic_matmul_transpose_a(
    %A: tensor<128x64xf32>, %B: tensor<128x64xf32>, %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<128x64xf32>, tensor<128x64xf32>)
    outs(%C : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_generic_matmul_transpose_a(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<128x64xf32>, %[[B:[0-9a-z]+]]: tensor<128x64xf32>, %[[C:[0-9a-z]+]]: tensor<64x64xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<2x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<128x64xf32> -> tensor<2x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<4x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<128x64xf32> -> tensor<4x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<2x4x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<64x64xf32> -> tensor<2x4x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<2x2x32x64xf32>, tensor<4x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<2x4x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<2x4x32x16xf32> -> tensor<64x64xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<64x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @block_generic_matmul_transpose_b(
    %A: tensor<64x128xf32>, %B: tensor<64x128xf32>, %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B : tensor<64x128xf32>, tensor<64x128xf32>)
    outs(%C : tensor<64x64xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_generic_matmul_transpose_b(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<64x128xf32>, %[[B:[0-9a-z]+]]: tensor<64x128xf32>, %[[C:[0-9a-z]+]]: tensor<64x64xf32>
// CHECK: %[[PACK_DST_0:.+]] = tensor.empty() : tensor<2x2x32x64xf32>
// CHECK: %[[A_PACKED:.+]] = linalg.pack %[[A]]
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[PACK_DST_0]] : tensor<64x128xf32> -> tensor<2x2x32x64xf32>
// CHECK: %[[PACK_DST_1:.+]] = tensor.empty() : tensor<4x2x16x64xf32>
// CHECK: %[[B_PACKED:.+]] = linalg.pack %[[B]]
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 64]
// CHECK-SAME:  into %[[PACK_DST_1]] : tensor<64x128xf32> -> tensor<4x2x16x64xf32>
// CHECK: %[[PACK_DST_2:.+]] = tensor.empty() : tensor<2x4x32x16xf32>
// CHECK: %[[C_PACKED:.+]] = linalg.pack %[[C]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[PACK_DST_2]] : tensor<64x64xf32> -> tensor<2x4x32x16xf32>
// CHECK: %[[GEMM_RES_PACKED:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A_PACKED]], %[[B_PACKED]] : tensor<2x2x32x64xf32>, tensor<4x2x16x64xf32>) outs(%[[C_PACKED]] : tensor<2x4x32x16xf32>)
// CHECK: %[[RES_UNPACKED:.+]] = linalg.unpack %[[GEMM_RES_PACKED]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[C]] : tensor<2x4x32x16xf32> -> tensor<64x64xf32>
// CHECK: return %[[RES_UNPACKED]] : tensor<64x64xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @non_contraction_generic(
    %A: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %c0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]}
    outs(%A : tensor<64x128xf32>) {
  ^bb0(%out: f32):
    %1 = arith.maximumf %out, %c0 : f32
    linalg.yield %1 : f32
  } -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @non_contraction_generic(
// CHECK-SAME:    %[[A:[0-9a-z]+]]: tensor<64x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NOT: linalg.pack
// CHECK: %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$MAP]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  outs(%[[A]] : tensor<64x128xf32>)
// CHECK-NOT: linalg.unpack
// CHECK: return %[[GENERIC]] : tensor<64x128xf32>

// RUN: mlir-opt %s -linalg-pack-matmul=block-factors=32,16,64 -canonicalize -split-input-file | FileCheck %s

func.func @block_matmul(
    %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x2x32x64xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x2x32x64xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<8x2x64x16xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 16]
// CHECK-SAME:  into %[[BUF1]] : tensor<128x128xf32> -> tensor<8x2x64x16xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x8x32x16xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[PACK2]] : tensor<4x8x32x16xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG2]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_constant(
    %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant dense<0.0> : tensor<128x128xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%cst : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_constant(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[BUF_RES:.+]] = arith.constant dense<0.000000e+00> : tensor<4x8x32x16xf32>
// CHECK-DAG: %[[BUF_OUT:.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[BUF_RES]] : tensor<4x8x32x16xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[BUF_OUT]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_producer(
    %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_producer(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BUF_RES:.+]] = tensor.empty() : tensor<4x8x32x16xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[BUF_RES]] : tensor<4x8x32x16xf32>) -> tensor<4x8x32x16xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  ins({{.*}} : tensor<4x2x32x64xf32>, tensor<8x2x64x16xf32>) outs(%[[FILL]] : tensor<4x8x32x16xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG2]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>

// -----

func.func @block_matmul_with_consumer(
    %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %2 = linalg.add ins(%1, %arg3 : tensor<128x128xf32>, tensor<128x128xf32>)
                  outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %2 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_matmul_with_consumer(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>, %[[ARG3:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-DAG: %[[BUF:.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  outs({{.*}} : tensor<4x8x32x16xf32>)
// CHECK: %[[UNPACK:.+]] = tensor.unpack %[[VAL]]
// CHECK-SAME:  inner_dims_pos = [0, 1] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG2]] : tensor<4x8x32x16xf32> -> tensor<128x128xf32>
// CHECK: %[[OUT:.+]] = linalg.add
// CHECK-SAME:  ins(%[[UNPACK]], %[[ARG3]] : tensor<128x128xf32>, tensor<128x128xf32>) outs(%[[BUF]] : tensor<128x128xf32>)
// CHECK: return %[[OUT]] : tensor<128x128xf32>

// -----

func.func @block_batch_matmul(
    %arg0: tensor<512x64x128xf32>, %arg1: tensor<512x128x64xf32>, %arg2: tensor<512x64x64xf32>) -> tensor<512x64x64xf32> {
  %0 = tensor.empty() : tensor<512x64x64xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x64x128xf32>, tensor<512x128x64xf32>)
                           outs(%arg2 : tensor<512x64x64xf32>) -> tensor<512x64x64xf32>
  return %1 : tensor<512x64x64xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>

// CHECK-LABEL: func @block_batch_matmul(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<512x64x128xf32>, %[[ARG1:.+]]: tensor<512x128x64xf32>, %[[ARG2:.+]]: tensor<512x64x64xf32>
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<512x2x2x32x64xf32>
// CHECK: %[[PACK0:.+]] = tensor.pack %[[ARG0]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 64]
// CHECK-SAME:  into %[[BUF0]] : tensor<512x64x128xf32> -> tensor<512x2x2x32x64xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<512x4x2x64x16xf32>
// CHECK: %[[PACK1:.+]] = tensor.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [0, 2, 1] inner_dims_pos = [1, 2] inner_tiles = [64, 16]
// CHECK-SAME:  into %[[BUF1]] : tensor<512x128x64xf32> -> tensor<512x4x2x64x16xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<512x2x4x32x16xf32>
// CHECK: %[[PACK2:.+]] = tensor.pack %[[ARG2]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[BUF2]] : tensor<512x64x64xf32> -> tensor<512x2x4x32x16xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<512x2x2x32x64xf32>, tensor<512x4x2x64x16xf32>) outs(%[[PACK2]] : tensor<512x2x4x32x16xf32>)
// CHECK: %[[OUT:.+]] = tensor.unpack %[[VAL]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 16]
// CHECK-SAME:  into %[[ARG2]] : tensor<512x2x4x32x16xf32> -> tensor<512x64x64xf32>
// CHECK: return %[[OUT]] : tensor<512x64x64xf32>

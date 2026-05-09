// RUN: mlir-opt %s --test-linalg-transform-patterns="test-decompose-local-softmax" | FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @local_softmax_decompose
// CHECK-SAME:    %[[INPUT:.*]]: tensor<4x128xf32>
// CHECK-SAME:    %[[OUTPUT:.*]]: tensor<4x4x32xf32>

// Step 1: Reshape input [4, 128] -> [4, 4, 32]
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INPUT]] {{\[\[}}0], [1, 2]]
// CHECK-SAME: tensor<4x128xf32> into tensor<4x4x32xf32>

// Step 2: Per-tile max reduction along ts (dim 2)
// CHECK: %[[MAX_INIT:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4xf32>)
// CHECK: %[[MAX:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[EXPANDED]] : tensor<4x4x32xf32>)
// CHECK:   arith.maxnumf
// CHECK:   -> tensor<4x4xf32>

// Step 3: exp(input - max)
// CHECK: %[[EXP:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[EXPANDED]], %[[MAX]] : tensor<4x4x32xf32>, tensor<4x4xf32>)
// CHECK:   arith.subf
// CHECK:   math.exp
// CHECK:   -> tensor<4x4x32xf32>

// Step 4: Per-tile sum reduction along ts (dim 2)
// CHECK: %[[SUM_INIT:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<4x4xf32>)
// CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[EXP]] : tensor<4x4x32xf32>)
// CHECK:   arith.addf
// CHECK:   -> tensor<4x4xf32>

// Step 5: P = exp(input - max) / sum
// CHECK: %[[P:.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[EXP]], %[[SUM]] : tensor<4x4x32xf32>, tensor<4x4xf32>)
// CHECK:   arith.divf
// CHECK:   -> tensor<4x4x32xf32>

// CHECK: return %[[P]], %[[MAX]], %[[SUM]]
func.func @local_softmax_decompose(%input : tensor<4x128xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

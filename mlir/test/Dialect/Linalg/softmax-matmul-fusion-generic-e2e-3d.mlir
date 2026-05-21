// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention (3D batched) using ONLY linalg.generic ops.
// Shapes: Q=[32,4,16], K^T=[32,16,128], V=[32,128,64], O=[32,4,64]
// Batch dimension (32) is fully parallel across all ops.

// CHECK-LABEL: func.func @flash_attention_3d_batch
// CHECK: linalg.batch_matmul
// CHECK: tensor.expand_shape
// Outer loop over batch (0 to 32), inner loop over tn (0 to 4):
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     linalg.generic
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK: return

func.func @flash_attention_3d_batch(%Q : tensor<32x4x16xf32>, %K_T : tensor<32x16x128xf32>, %V : tensor<32x128x64xf32>) -> tensor<32x4x64xf32> {
  %S_init = tensor.empty() : tensor<32x4x128xf32>
  %S = linalg.batch_matmul ins(%Q, %K_T : tensor<32x4x16xf32>, tensor<32x16x128xf32>) outs(%S_init : tensor<32x4x128xf32>) -> tensor<32x4x128xf32>
  %softmax_init = tensor.empty() : tensor<32x4x128xf32>
  %softmax = linalg.softmax dimension(2) ins(%S : tensor<32x4x128xf32>) outs(%softmax_init : tensor<32x4x128xf32>) -> tensor<32x4x128xf32>
  %O_init = tensor.empty() : tensor<32x4x64xf32>
  %O = linalg.batch_matmul ins(%softmax, %V : tensor<32x4x128xf32>, tensor<32x128x64xf32>) outs(%O_init : tensor<32x4x64xf32>) -> tensor<32x4x64xf32>
  return %O : tensor<32x4x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // Match the 3D rescaling matmul: (batch, m, tn, ts, kv)
    %rescaling = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<parallel>
        ]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Tile both batch (dim 0) and tn (dim 2) dimensions
    %fused, %loops:2 = transform.structured.fuse %rescaling tile_sizes [1, 0, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

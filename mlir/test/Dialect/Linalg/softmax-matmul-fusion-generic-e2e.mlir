// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention (2D) using ONLY linalg.generic ops.
// See softmax-matmul-fusion-generic-e2e-3d.mlir for the batched (3D) case.
//
// NOTE: The first GEMM is not fused into the loop because expand_shape blocks
// producer fusion. See design doc for alternatives.

// CHECK-LABEL: func.func @flash_attention_2d
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape
// CHECK: scf.for
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK: return

func.func @flash_attention_2d(%Q : tensor<4x16xf32>, %K_T : tensor<16x128xf32>, %V : tensor<128x64xf32>) -> tensor<4x64xf32> {
  %S_init = tensor.empty() : tensor<4x128xf32>
  %S = linalg.matmul ins(%Q, %K_T : tensor<4x16xf32>, tensor<16x128xf32>) outs(%S_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%S : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %O_init = tensor.empty() : tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %O : tensor<4x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %rescaling = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<parallel>
        ]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %loop = transform.structured.fuse %rescaling tile_sizes [0, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention using ONLY linalg.generic ops (no linalg.local_softmax).
//
// After rewrite + tile-and-fuse:
// - Local softmax generics (max, exp, sum, div) are fused inside the scf.for loop
// - The rescaling matmul generic is tiled inside the loop
// - The first GEMM remains outside (expand_shape prevents auto-fusion)
//
// NOTE: The first GEMM is not fused into the loop because expand_shape blocks
// producer fusion in the current infrastructure. To fully fuse the first GEMM,
// either:
// (a) Use the named linalg.local_softmax op (see online-softmax branch), or
// (b) Fold the expand_shape into the generic indexing maps, or
// (c) Write a dedicated pass using tileAndFuseProducerOfSlice with bubble-up.
//
// What IS demonstrated: the local softmax computation (4 generics) tiles and
// fuses correctly into the rescaling matmul's tile loop via structured.fuse.

// CHECK-LABEL: func.func @flash_attention_generic_e2e
// The first GEMM and expand_shape remain outside the loop:
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape
// The scf.for loop contains all local softmax generics + rescaling matmul:
// CHECK: scf.for
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK: return

func.func @flash_attention_generic_e2e(%Q : tensor<4x16xf32>, %K_T : tensor<16x128xf32>, %V : tensor<128x64xf32>) -> tensor<4x64xf32> {
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
    // Use transform.structured.fuse to tile the rescaling matmul and
    // auto-fuse its direct producers (the local softmax generics).
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

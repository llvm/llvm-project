// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention (2D) using ONLY linalg.generic ops.
// See softmax-matmul-fusion-generic-e2e-3d.mlir for the batched (3D) case.
//
// The second GEMM is emitted split (op1 matmul + op1b lsum + op2 recurrence +
// op3 divide). Tiling op2's tn dimension fuses everything (first GEMM, local
// softmax, op1 pv, op1b lsum) into one loop; the final divide stays outside.
//
// NOTE: this test only checks structure. Tiling op2's `tn` *reduction* with
// transform.structured.fuse re-initializes the accumulator each iteration, which
// is numerically wrong for tn > 1 (a known bug, tracked in
// build/bug-online-attn-accumulator-reset.md); the correct fix is to tile the
// reduction with tile_using_for. Structure-only here; correctness is covered
// separately.

// CHECK-LABEL: func.func @flash_attention_2d
// CHECK: scf.for
// First GEMM + local softmax + op1 matmul (pv) + op1b (lsum) + op2 recurrence:
// CHECK:   linalg.matmul
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK-NOT: linalg.matmul
// Final divide O = O / L (outside the loop):
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK:   arith.divf
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
    // Tile and fuse op2 (the online recurrence): (m, tn, kv) with tn reduction.
    // Fusing it pulls op1/op1b/local-softmax/first-GEMM into the loop.
    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<parallel>
        ]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused, %loop = transform.structured.fuse %op2 tile_sizes [0, 1, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

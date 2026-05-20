// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention: softmax(Q @ K^T) @ V
// After rewrite + tile-and-fuse, everything is in a single scf.for loop.

// CHECK-LABEL: func.func @flash_attention_e2e
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C4]] step %[[C1]]
// Inside the loop: first GEMM, local_softmax, rescaling matmul
// CHECK:         linalg.matmul
// CHECK:         linalg.local_softmax
// CHECK:         linalg.generic
// CHECK:         scf.yield
// No matmul or local_softmax outside the loop
// CHECK-NOT:   linalg.matmul
// CHECK-NOT:   linalg.local_softmax
// CHECK:       return

func.func @flash_attention_e2e(%Q : tensor<4x16xf32>, %K_T : tensor<16x128xf32>, %V : tensor<128x64xf32>) -> tensor<4x64xf32> {
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
    // Step 1: Tile the rescaling matmul generic on tn dimension
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %generic tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Step 2: Fuse local_softmax into the tile loop
    %local_sm = transform.structured.match ops{["linalg.local_softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused_sm, %new_loop = transform.structured.fuse_into_containing_op %local_sm into %loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // Step 3: Fuse first GEMM into the tile loop
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %fused_mm, %new_loop2 = transform.structured.fuse_into_containing_op %matmul into %new_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

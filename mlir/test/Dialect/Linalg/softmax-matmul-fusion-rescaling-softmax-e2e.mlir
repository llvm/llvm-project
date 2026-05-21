// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end test: softmax with multiple users.
// The softmax result feeds both a matmul AND is returned directly.
// After rewrite + tile-and-fuse:
// - The rescaling matmul is tiled and fused with local softmax generics (scf.for)
// - The rescaling softmax (2 generics + collapse_shape) recovers global softmax

// CHECK-LABEL: func.func @softmax_multi_user_e2e
// The rescaling matmul loop (fused with local softmax):
// CHECK: scf.for
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   linalg.generic
// CHECK:   scf.yield
//
// The rescaling softmax (recover global softmax for the other user):
// Generic 1: reduce over tn
// CHECK: linalg.generic
// CHECK-SAME: "reduction"
// Generic 2: elementwise correction
// CHECK: linalg.generic
// CHECK: tensor.collapse_shape
// CHECK-SAME: into tensor<4x128xf32>
// CHECK: return

func.func @softmax_multi_user_e2e(%input : tensor<4x128xf32>, %V : tensor<128x64xf32>) -> (tensor<4x64xf32>, tensor<4x128xf32>) {
  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%input : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %O_init = tensor.empty() : tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %O, %softmax : tensor<4x64xf32>, tensor<4x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // Tile and fuse the rescaling matmul on tn dimension
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

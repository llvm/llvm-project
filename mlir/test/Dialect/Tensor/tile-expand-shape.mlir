// RUN: mlir-opt %s --split-input-file --test-linalg-transform-patterns --transform-interpreter --canonicalize --cse | FileCheck %s

// Test that tensor.expand_shape implements TilingInterface and can be tiled.

// CHECK-LABEL: func.func @tile_expand_shape
// CHECK: scf.for
// CHECK:   tensor.extract_slice
// CHECK:   tensor.expand_shape
// CHECK:   scf.yield
// CHECK: return
func.func @tile_expand_shape(%input : tensor<4x128xf32>) -> tensor<4x4x32xf32> {
  %expanded = tensor.expand_shape %input [[0], [1, 2]] output_shape [4, 4, 32]
      : tensor<4x128xf32> into tensor<4x4x32xf32>
  return %expanded : tensor<4x4x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %expand = transform.structured.match ops{["tensor.expand_shape"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Tile along the tn dimension (dim 1 of the expanded output)
    %tiled, %loop = transform.structured.tile_using_for %expand tile_sizes [0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Test that expand_shape can be fused as a producer into a consumer's tile loop.

// CHECK-LABEL: func.func @fuse_through_expand_shape
// CHECK: scf.for
// CHECK:   tensor.extract_slice %{{.*}} : tensor<4x128xf32> to tensor<4x32xf32>
// CHECK:   tensor.expand_shape
// CHECK:   linalg.generic
// CHECK:   scf.yield
// CHECK: return
func.func @fuse_through_expand_shape(%input : tensor<4x128xf32>) -> tensor<4x4xf32> {
  %expanded = tensor.expand_shape %input [[0], [1, 2]] output_shape [4, 4, 32]
      : tensor<4x128xf32> into tensor<4x4x32xf32>
  %init = tensor.empty() : tensor<4x4xf32>
  %cst = arith.constant 0.0 : f32
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  // Reduce over the ts dimension (dim 2) — parallel over tn (dim 1)
  %result = linalg.generic {
      indexing_maps = [affine_map<(m, tn, ts) -> (m, tn, ts)>,
                       affine_map<(m, tn, ts) -> (m, tn)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%expanded : tensor<4x4x32xf32>) outs(%filled : tensor<4x4xf32>) {
    ^bb0(%in : f32, %out : f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
    } -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Tile the tn dimension — this should fuse expand_shape as a producer
    %fused, %loop = transform.structured.fuse %generic tile_sizes [0, 1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

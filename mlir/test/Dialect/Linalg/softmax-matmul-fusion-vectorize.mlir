// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --transform-interpreter \
// RUN:   --canonicalize --cse | FileCheck %s

// End-to-end FlashAttention (3D batched) softmax-matmul-fusion, followed by
// vectorization of the tile-and-fused loop. Based on
// softmax-matmul-fusion-generic-e2e-3d.mlir.
// Shapes: Q=[32,4,16], K^T=[32,16,128], V=[32,128,64], O=[32,4,64].
//
// Goal: the fused loop should FULLY vectorize. Because the second GEMM is
// emitted split (op1 = pure contraction over ts, op2 = online recurrence over
// tn), BOTH matmuls now lower to vector.contract:
//   - first GEMM  (Q*K^T)  -> vector.contract
//   - op1 (num*V per tile) -> vector.contract
// op2's `tn` reduction is tiled to 1 and folded away with
// fold_unit_extent_dims_via_slices, so the recurrence becomes pure elementwise
// (max/exp/mul/add) and vectorizes to arith + vector.multi_reduction. No
// linalg.generic survives in the loop body.

// CHECK-LABEL: func.func @flash_vectorize
// CHECK: scf.for
// CHECK:   scf.for
// First GEMM (Q*K^T) fused into the loop -> vector.contract:
// CHECK:     vector.contract
// CHECK-SAME: into vector<1x4x32xf32>
// op1 (num*V per tile) -> vector.contract (the second GEMM, now vectorized):
// CHECK:     vector.contract
// CHECK-SAME: into vector<4x64xf32>
// op1b denominator sum over ts -> vector.multi_reduction <add>:
// CHECK:     vector.multi_reduction <add>
// op2 recurrence is pure elementwise after fold -> no linalg.generic:
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK-NOT: linalg.generic
// CHECK-NOT: linalg.batch_matmul
// Final divide O = O / L, also vectorized:
// CHECK: arith.divf {{.*}} vector
// CHECK: return

func.func @flash_vectorize(%Q : tensor<32x4x16xf32>, %K_T : tensor<32x16x128xf32>, %V : tensor<32x128x64xf32>) -> tensor<32x4x64xf32> {
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
    // Match op2 (the online recurrence): (batch, m, tn, kv) with tn reduction.
    %op2 = transform.structured.match ops{["linalg.generic"]}
        attributes{iterator_types = [
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<parallel>,
          #linalg.iterator_type<reduction>,
          #linalg.iterator_type<parallel>
        ]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // Tile batch (dim 0) and tn (dim 2) to 1; fusing op2 pulls op1/op1b/local
    // softmax and the first GEMM into the loops.
    %fused, %loops:2 = transform.structured.fuse %op2 tile_sizes [1, 0, 1, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // Drop the unit tn reduction dim so the recurrence becomes pure elementwise.
    %func = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    } : !transform.any_op
    // Vectorize the whole function: both GEMMs -> vector.contract; the
    // elementwise/reduction generics -> vector ops.
    %func2 = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %func_v = transform.structured.vectorize_children_and_apply_patterns %func2
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

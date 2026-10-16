// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

// `arith.mulf` + `vector.multi_reduction` matches what `linalg.matmul` produces
// after `transform.structured.vectorize` (with matmul-style transfer_read layout).
// This checks the follow-on stack: multi_reduction → contract → (transfer layout
// cleanup) → outer-product-style contraction lowering → `vector.fma`.

#map_lhs = affine_map<(d0, d1) -> (d0, 0, d1)>
#map_rhs = affine_map<(d0, d1) -> (0, d1, d0)>

// CHECK-LABEL: func @multi_reduction_to_fma
// CHECK-SAME: memref<3x4xf32>
// CHECK-SAME: memref<4x3xf32>
// CHECK-SAME: memref<3x3xf32>
// CHECK-DAG: vector.transfer_read {{.*}} : memref<3x4xf32>, vector<3x4xf32>
// CHECK-DAG: vector.transfer_read {{.*}} : memref<4x3xf32>, vector<4x3xf32>
// CHECK-DAG: vector.transfer_read {{.*}} : memref<3x3xf32>, vector<3x3xf32>
// One dot-product row uses three fused multiply-adds along K; 3 output rows × 4 K steps.
// CHECK-COUNT-12: vector.fma
// CHECK-NOT: vector.multi_reduction
// CHECK-NOT: vector.contract
// CHECK-NOT: vector.outerproduct
// CHECK: vector.transfer_write {{.*}} : vector<3x3xf32>, memref<3x3xf32>
func.func @multi_reduction_to_fma(%A: memref<3x4xf32>, %B: memref<4x3xf32>, %C: memref<3x3xf32>) {
  %c0 = arith.constant 0 : index
  %p = ub.poison : f32
  %va = vector.transfer_read %A[%c0, %c0], %p {permutation_map = #map_lhs} : memref<3x4xf32>, vector<3x3x4xf32>
  %vb = vector.transfer_read %B[%c0, %c0], %p {permutation_map = #map_rhs} : memref<4x3xf32>, vector<3x3x4xf32>
  %vc = vector.transfer_read %C[%c0, %c0], %p : memref<3x3xf32>, vector<3x3xf32>
  %mul = arith.mulf %va, %vb : vector<3x3x4xf32>
  %acc = vector.multi_reduction <add>, %mul, %vc [2] : vector<3x3x4xf32> to vector<3x3xf32>
  vector.transfer_write %acc, %C[%c0, %c0] : vector<3x3xf32>, memref<3x3xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.transfer_permutation_patterns
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.any_op
    transform.yield
  }
}

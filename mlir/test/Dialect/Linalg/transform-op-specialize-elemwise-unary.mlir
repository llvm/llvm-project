// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @specialize_exp(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%arg0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.exp %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: specialize_exp
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[ARG0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

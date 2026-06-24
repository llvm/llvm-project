// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @specialize_exp(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%arg0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.exp %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %1 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.log %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %2 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%1 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.absf %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %3 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%2 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.ceil %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %4 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%3 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.floor %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %5 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%4 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.negf %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %cst_1 = arith.constant 1.0 : f32
  %6 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%5 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.divf %cst_1, %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %7 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%6 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.round %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %8 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%7 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.sqrt %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %9 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%8 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.rsqrt %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %10 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%9 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = arith.mulf %in, %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %11 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%10 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.tanh %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  %12 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%11 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %v = math.erf %in : f32
    linalg.yield %v : f32
  } -> tensor<?x?x?xf32>
  return %12 : tensor<?x?x?xf32>
}
// CHECK-LABEL: specialize_exp
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: %[[RES0:.+]] = linalg.exp ins(%[[ARG0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES1:.+]] = linalg.log ins(%[[RES0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES2:.+]] = linalg.abs ins(%[[RES1]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES3:.+]] = linalg.ceil ins(%[[RES2]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES4:.+]] = linalg.floor ins(%[[RES3]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES5:.+]] = linalg.negf ins(%[[RES4]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES6:.+]] = linalg.reciprocal ins(%[[RES5]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES7:.+]] = linalg.round ins(%[[RES6]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES8:.+]] = linalg.sqrt ins(%[[RES7]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES9:.+]] = linalg.rsqrt ins(%[[RES8]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES10:.+]] = linalg.square ins(%[[RES9]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES11:.+]] = linalg.tanh ins(%[[RES10]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK: %[[RES12:.+]] = linalg.erf ins(%[[RES11]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// RUN: mlir-opt %s --test-transform-dialect-interpreter --split-input-file | FileCheck %s

// CHECK-DAG: #[[$MAP13:.+]] = affine_map<() -> (13)>

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    transform.structured.multitile_sizes %0 { target_size = 3, dimension = 0 }
}

// CHECK-LABEL: @multitile_sizes_static
func.func @multitile_sizes_static(
  %arg0: tensor<13x34xf32>, %arg1: tensor<34x42xf32>, %arg2: tensor<13x42xf32>)
    -> tensor<13x42xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<13x34xf32>, tensor<34x42xf32>)
                     outs(%arg2: tensor<13x42xf32>)
    -> tensor<13x42xf32>
  // The first application computes the total size.
  // CHECK: %{{.*}} = affine.apply #[[$MAP13]]()
  // CHECK: %[[SIZE:.+]] = affine.apply #[[$MAP13]]()
  // CHECK: %[[COND:.+]] = arith.cmpi eq, %[[SIZE]], %{{.*}}
  // CHECK: cf.assert %[[COND]], "could not compute dynamic multi-size tile shapes"

  return %0 : tensor<13x42xf32>
}

// -----

transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    transform.structured.multitile_sizes %0 { target_size = 3, divisor = 2, dimension = 0 }
}

// CHECK: #[[$MAP_A:.+]] = affine_map<()[s0] -> ([[A_IMPL:s0 floordiv 2]])>
// CHECK: #[[$MAP_T:.+]] = affine_map<() -> (2)>
// CHECK: #[[$MAP_D:.+]] = affine_map<()[s0] -> ([[D_IMPL:\(s0 floordiv 2 \+ 1\) floordiv 2]])>
// CHECK: #[[$MAP_S:.+]] = affine_map<()[s0] -> ((([[A_IMPL]]) floordiv ([[D_IMPL]])) * 2)>
// CHECK: #[[$MAP_V:.+]] = affine_map<()[s0] -> (([[A_IMPL]]) mod ([[D_IMPL]]))>
// CHECK: #[[$MAP_U:.+]] = affine_map<()[s0] -> ([[D_IMPL]] - ([[A_IMPL]]) mod ([[D_IMPL]]))>

// CHECK-LABEL: @multitile_sizes_dynamic
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>, %{{.*}}: tensor<?x?xf32>, %{{.*}}: tensor<?x?xf32>)
func.func @multitile_sizes_dynamic(
  // For matmul, the extent of the first iteration space dimension is equal to
  // the size of the first dimension of the first tensor. The indexing map was
  // folded so there is no map application happening.
  //
  // CHECK: %[[C0:.+]] = arith.constant 0
  // CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]]
  //
  // The following are the maps as emitted by computeMultiTileSizes.
  // CHECK: affine.apply #[[$MAP_A]]()[%[[DIM]]]
  // CHECK: affine.apply #[[$MAP_T]]()
  // CHECK: affine.apply #[[$MAP_D]]()[%[[DIM]]]
  // CHECK: affine.apply #[[$MAP_S]]()[%[[DIM]]]
  // CHECK: affine.apply #[[$MAP_V]]()[%[[DIM]]]
  // CHECK: affine.apply #[[$MAP_U]]()[%[[DIM]]]
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
}

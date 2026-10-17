// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops -linalg-morph-ops=generic-to-category | FileCheck %s

#projection = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>
#identity   = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @transpose_and_broadcast(%x : tensor<7x8x9xf32>, %y:  tensor<5x9x7x8x10xf32>, %z :  tensor<5x9x7x8x10xf32>) ->  tensor<5x9x7x8x10xf32> {
  %res = linalg.generic
     { indexing_maps = [#projection, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<7x8x9xf32>, tensor<5x9x7x8x10xf32>) outs(%z : tensor<5x9x7x8x10xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<5x9x7x8x10xf32>
  return %res : tensor<5x9x7x8x10xf32>
}

// CHECK-LABEL: transpose_and_broadcast
// CHECK-SAME: %[[X:.+]]: tensor<7x8x9xf32>, %[[Y:.+]]: tensor<5x9x7x8x10xf32>, %[[Z:.+]]: tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32> {
// CHECK: {{.*}} = linalg.elementwise <div> indexing_maps = {{.*}} ins(%[[X]], %[[Y]] : tensor<7x8x9xf32>, tensor<5x9x7x8x10xf32>) outs(%[[Z]] : tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transposed = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

func.func @transpose_only(%x : tensor<32x2x16xf32>, %y:  tensor<2x16x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
  %res = linalg.generic
     { indexing_maps = [#transposed, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<32x2x16xf32>, tensor<2x16x32xf32>)
     outs(%z : tensor<2x16x32xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<2x16x32xf32>
  return %res : tensor<2x16x32xf32>
}

// CHECK-LABEL: transpose_only
// CHECK-SAME: %[[X:.+]]: tensor<32x2x16xf32>, %[[Y:.+]]: tensor<2x16x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: {{.*}} = linalg.elementwise <div> indexing_maps = {{.*}} ins(%[[X]], %[[Y]] : tensor<32x2x16xf32>, tensor<2x16x32xf32>) outs(%[[Z]] : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @broadcast_only(%x : tensor<2x16x32xf32>, %y:  tensor<2x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
  %res = linalg.generic
     { indexing_maps = [#identity, #broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
     ins(%x, %y : tensor<2x16x32xf32>, tensor<2x32xf32>)
     outs(%z : tensor<2x16x32xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<2x16x32xf32>
  return %res : tensor<2x16x32xf32>
}

// CHECK-LABEL: broadcast_only
// CHECK-SAME: %[[X:.+]]: tensor<2x16x32xf32>, %[[Y:.+]]: tensor<2x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: {{.*}} = linalg.elementwise <div> indexing_maps = {{.*}} ins(%[[X]], %[[Y]] : tensor<2x16x32xf32>, tensor<2x32xf32>) outs(%arg2 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic

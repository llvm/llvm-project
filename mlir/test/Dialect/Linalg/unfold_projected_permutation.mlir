// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#projection = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d1)>
#identity   = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @test_mixed(%x : tensor<7x8x9xf32>, %y:  tensor<5x9x7x8x10xf32>, %z :  tensor<5x9x7x8x10xf32>) ->  tensor<5x9x7x8x10xf32> {
  %res = linalg.generic
     { indexing_maps = [#projection, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} 
     ins(%x, %y : tensor<7x8x9xf32>, tensor<5x9x7x8x10xf32>) outs(%z : tensor<5x9x7x8x10xf32>) {
     ^bb0(%in: f32, %in_1: f32, %out: f32):
       %div = arith.divf %in, %in_1 : f32
       linalg.yield %div : f32
  } -> tensor<5x9x7x8x10xf32>
  return %res : tensor<5x9x7x8x10xf32>
}

// CHECK-LABEL: test_mixed
// CHECK-SAME: %[[X:.+]]: tensor<7x8x9xf32>, %[[Y:.+]]: tensor<5x9x7x8x10xf32>, %[[Z:.+]]: tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<9x7x8xf32>
// CHECK: %[[Transposed:.+]] = linalg.transpose ins(%[[X]] : tensor<7x8x9xf32>) outs(%[[E0]] : tensor<9x7x8xf32>) permutation = [2, 0, 1]
// CHECK: %[[E1:.+]] = tensor.empty() : tensor<5x9x7x8x10xf32>
// CHECK: %[[Broadcasted:.+]] = linalg.broadcast ins(%[[Transposed]] : tensor<9x7x8xf32>) outs(%[[E1]] : tensor<5x9x7x8x10xf32>) dimensions = [0, 4]
// CHECK: {{.*}} = linalg.div ins(%[[Broadcasted]], %[[Y]] : tensor<5x9x7x8x10xf32>, tensor<5x9x7x8x10xf32>) outs(%[[Z]] : tensor<5x9x7x8x10xf32>) -> tensor<5x9x7x8x10xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transposed = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

func.func @test_transposed(%x : tensor<32x2x16xf32>, %y:  tensor<2x16x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
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

// CHECK-LABEL: test_transposed
// CHECK-SAME: %[[X:.+]]: tensor<32x2x16xf32>, %[[Y:.+]]: tensor<2x16x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK: %[[Transposed:.+]] = linalg.transpose ins(%[[X]] : tensor<32x2x16xf32>) outs(%[[E0]] : tensor<2x16x32xf32>) permutation = [1, 2, 0]
// CHECK: {{.*}} = linalg.div ins(%[[Transposed]], %[[Y]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) outs(%[[Z]] : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic

// -----

#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d0, d2)>
func.func @test_broadcast(%x : tensor<2x16x32xf32>, %y:  tensor<2x32xf32>, %z :  tensor<2x16x32xf32>) ->  tensor<2x16x32xf32> {
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

// CHECK-LABEL: test_broadcast
// CHECK-SAME: %[[X:.+]]: tensor<2x16x32xf32>, %[[Y:.+]]: tensor<2x32xf32>, %[[Z:.+]]: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
// CHECK: %[[E0:.+]] = tensor.empty() : tensor<2x16x32xf32>
// CHECK: %[[Broadcasted:.+]] = linalg.broadcast ins(%[[Y]] : tensor<2x32xf32>) outs(%[[E0]] : tensor<2x16x32xf32>) dimensions = [1]
// CHECK: {{.*}} = linalg.div ins(%[[X]], %[[Broadcasted]] : tensor<2x16x32xf32>, tensor<2x16x32xf32>) outs(%arg2 : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
// CHECK-NOT: linalg.generic

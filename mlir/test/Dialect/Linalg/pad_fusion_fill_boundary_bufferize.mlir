// RUN: mlir-opt %s -test-linalg-pad-fusion=fill-boundary-only \
// RUN:     -one-shot-bufferize="bufferize-function-boundaries" -cse -canonicalize \
// RUN:   > %t
// RUN: FileCheck %s < %t
// RUN: FileCheck %s --check-prefix=ONEALLOC < %t
// RUN: FileCheck %s --check-prefix=NOCOPY < %t

// Separate prefixes: a CHECK-NOT would only cover the gap between two checks.
func.func @pad_fusion_boundary_bufferizes_in_place(
    %arg0 : tensor<4x3xf32>, %arg1 : f32) -> tensor<7x6xf32> {
  %init = tensor.empty() : tensor<4x3xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<4x3xf32>) outs(%init : tensor<4x3xf32>) {
    ^bb0(%arg2 : f32, %arg3 : f32):
      %1 = arith.mulf %arg2, %arg2 : f32
      linalg.yield %1 : f32
    } -> tensor<4x3xf32>
  %1 = tensor.pad %0 low [1, 2] high [2, 1] {
    ^bb0(%arg2: index, %arg3 : index):
      tensor.yield %arg1 : f32
    } : tensor<4x3xf32> to tensor<7x6xf32>
  return %1 : tensor<7x6xf32>
}

//      CHECK: func @pad_fusion_boundary_bufferizes_in_place
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: memref<4x3xf32
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: f32
//      CHECK:   %[[ALLOC:.+]] = memref.alloc() {{.*}} : memref<7x6xf32>
//      CHECK:   %[[TOP:.+]] = memref.subview %[[ALLOC]][0, 0] [1, 6] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32) outs(%[[TOP]]
//      CHECK:   %[[BOT:.+]] = memref.subview %[[ALLOC]][5, 0] [2, 6] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32) outs(%[[BOT]]
//      CHECK:   %[[LEFT:.+]] = memref.subview %[[ALLOC]][1, 0] [4, 2] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32) outs(%[[LEFT]]
//      CHECK:   %[[RIGHT:.+]] = memref.subview %[[ALLOC]][1, 5] [4, 1] [1, 1]
//      CHECK:   linalg.fill ins(%[[ARG1]] : f32) outs(%[[RIGHT]]
//      CHECK:   %[[INTERIOR:.+]] = memref.subview %[[ALLOC]][1, 2] [4, 3] [1, 1]
//      CHECK:   linalg.generic
// CHECK-SAME:       outs(%[[INTERIOR]]
//      CHECK:   return %[[ALLOC]]

// ONEALLOC-COUNT-1: memref.alloc
//     ONEALLOC-NOT: memref.alloc
//       NOCOPY-NOT: memref.copy

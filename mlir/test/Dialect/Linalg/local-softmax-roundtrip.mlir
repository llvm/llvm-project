// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: func.func @local_softmax_basic
// CHECK: linalg.local_softmax dimension(1) tile_size(32)
// CHECK-SAME: ins(%{{.*}} : tensor<4x128xf32>)
// CHECK-SAME: outs(%{{.*}} : tensor<4x4x32xf32>, %{{.*}} : tensor<4x4xf32>, %{{.*}} : tensor<4x4xf32>)
// CHECK-SAME: -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
func.func @local_softmax_basic(%input : tensor<4x128xf32>,
    %output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> (tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:3 = linalg.local_softmax dimension(1) tile_size(32)
    ins(%input : tensor<4x128xf32>)
    outs(%output : tensor<4x4x32xf32>, %max : tensor<4x4xf32>, %den : tensor<4x4xf32>)
    -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  return %0#0, %0#1, %0#2 : tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @local_softmax_dim0
// CHECK: linalg.local_softmax dimension(0) tile_size(16)
func.func @local_softmax_dim0(%input : tensor<64x32xf32>,
    %output : tensor<4x16x32xf32>, %max : tensor<4x32xf32>, %den : tensor<4x32xf32>)
    -> (tensor<4x16x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>) {
  %0:3 = linalg.local_softmax dimension(0) tile_size(16)
    ins(%input : tensor<64x32xf32>)
    outs(%output : tensor<4x16x32xf32>, %max : tensor<4x32xf32>, %den : tensor<4x32xf32>)
    -> tensor<4x16x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>
  return %0#0, %0#1, %0#2 : tensor<4x16x32xf32>, tensor<4x32xf32>, tensor<4x32xf32>
}

// -----

// CHECK-LABEL: func.func @local_softmax_3d
// CHECK: linalg.local_softmax dimension(2) tile_size(64)
func.func @local_softmax_3d(%input : tensor<2x8x256xf32>,
    %output : tensor<2x8x4x64xf32>, %max : tensor<2x8x4xf32>, %den : tensor<2x8x4xf32>)
    -> (tensor<2x8x4x64xf32>, tensor<2x8x4xf32>, tensor<2x8x4xf32>) {
  %0:3 = linalg.local_softmax dimension(2) tile_size(64)
    ins(%input : tensor<2x8x256xf32>)
    outs(%output : tensor<2x8x4x64xf32>, %max : tensor<2x8x4xf32>, %den : tensor<2x8x4xf32>)
    -> tensor<2x8x4x64xf32>, tensor<2x8x4xf32>, tensor<2x8x4xf32>
  return %0#0, %0#1, %0#2 : tensor<2x8x4x64xf32>, tensor<2x8x4xf32>, tensor<2x8x4xf32>
}

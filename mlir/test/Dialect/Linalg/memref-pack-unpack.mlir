// RUN: mlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_pack_memref
func.func @test_pack_memref(%arg0: memref<128x256xf32>, %arg1: memref<16x8x8x32xf32>) {
  // CHECK-NOT: %{{.*}} = linalg.pack
  // CHECK: linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %{{.*}} : memref<128x256xf32> -> memref<16x8x8x32xf32>
  linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : memref<128x256xf32> -> memref<16x8x8x32xf32>
  return
}

// -----

// CHECK-LABEL: func @test_unpack_memref
func.func @test_unpack_memref(%arg0: memref<16x8x8x32xf32>, %arg1: memref<128x256xf32>) {
  // CHECK: linalg.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %{{.*}} : memref<16x8x8x32xf32>
  linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : memref<16x8x8x32xf32> -> memref<128x256xf32>
  return
}

// -----

// CHECK-LABEL: func @test_pack_memref_with_padding
func.func @test_pack_memref_with_padding(%arg0: memref<127x255xf32>, %arg1: memref<16x8x8x32xf32>, %pad: f32) {
  // CHECK: linalg.pack %{{.*}} padding_value(%{{.*}} : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %{{.*}} : memref<127x255xf32>
  linalg.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : memref<127x255xf32> -> memref<16x8x8x32xf32>
  return
}

// -----

// CHECK-LABEL: func @test_pack_tensor
func.func @test_pack_tensor(%arg0: tensor<128x256xf32>, %arg1: tensor<16x8x8x32xf32>) -> tensor<16x8x8x32xf32> {
  // CHECK: %[[RESULT:.*]] = linalg.pack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %{{.*}} : tensor<128x256xf32> -> tensor<16x8x8x32xf32>
  %0 = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : tensor<128x256xf32> -> tensor<16x8x8x32xf32>
  // CHECK: return %[[RESULT]] : tensor<16x8x8x32xf32>
  return %0 : tensor<16x8x8x32xf32>
}

// -----

// CHECK-LABEL: func @test_unpack_tensor
func.func @test_unpack_tensor(%arg0: tensor<16x8x8x32xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  // CHECK: %[[RESULT:.*]] = linalg.unpack %{{.*}} inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %{{.*}} : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
  %0 = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %arg1 : tensor<16x8x8x32xf32> -> tensor<128x256xf32>
  // CHECK: return %[[RESULT]] : tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}
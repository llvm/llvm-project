// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=cast}))" 2>&1 | FileCheck %s

func.func @test_memref_cast_identity() {
  %0 = memref.alloca() {test.cast, cast_dest = memref<f32>} : memref<f32>
  // CHECK: Successfully generated cast for operation: %[[V:.*]] = memref.alloca(){{.*}} : memref<f32>
  // CHECK: Cast result type: memref<f32>
  return
}

// -----

func.func @test_memref_cast_static_to_dynamic() {
  %0 = memref.alloca() {test.cast, cast_dest = memref<?xf32>} : memref<4xf32>
  // CHECK: Successfully generated cast for operation: %[[V:.*]] = memref.alloca(){{.*}} : memref<4xf32>
  // CHECK: Cast result type: memref<?xf32>
  // CHECK: Generated: %{{.*}} = memref.cast %[[V]] : memref<4xf32> to memref<?xf32>
  return
}

// -----

func.func @test_memref_memory_space_cast() {
  %0 = memref.alloca() {test.cast, cast_dest = memref<4xf32, 1>} : memref<4xf32>
  // CHECK: Successfully generated cast for operation: %[[V:.*]] = memref.alloca(){{.*}} : memref<4xf32>
  // CHECK: Cast result type: memref<4xf32, 1>
  // CHECK: Generated: %{{.*}} = memref.memory_space_cast %[[V]] : memref<4xf32> to memref<4xf32, 1>
  return
}

// -----

func.func @test_memref_cast_incompatible() {
  %0 = memref.alloca() {test.cast, cast_dest = tensor<4xf32>} : memref<4xf32>
  // CHECK: Failed to generate cast for operation: %{{.*}} = memref.alloca(){{.*}} : memref<4xf32>
  return
}

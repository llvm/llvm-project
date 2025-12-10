// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=load}))" 2>&1 | FileCheck %s

func.func @test_memref_load_scalar() {
  %ptr = memref.alloca() {test.ptr} : memref<f32>
  // CHECK: Successfully generated load for operation: %[[PTR:.*]] = memref.alloca() {test.ptr} : memref<f32>
  // CHECK: Loaded value type: f32
  // CHECK: Generated: %{{.*}} = memref.load %[[PTR]][] : memref<f32>
  return
}

// -----

func.func @test_memref_load_int() {
  %ptr = memref.alloca() {test.ptr} : memref<i64>
  // CHECK: Successfully generated load for operation: %[[PTR:.*]] = memref.alloca() {test.ptr} : memref<i64>
  // CHECK: Loaded value type: i64
  // CHECK: Generated: %{{.*}} = memref.load %[[PTR]][] : memref<i64>
  return
}

// -----

func.func @test_memref_load_dynamic() {
  %c10 = arith.constant 10 : index
  %ptr = memref.alloc(%c10) {test.ptr} : memref<?xf32>
  // CHECK: Failed to generate load for operation: %[[PTR:.*]] = memref.alloc(%{{.*}}) {test.ptr} : memref<?xf32>
  return
}


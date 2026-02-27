// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=store}))" 2>&1 | FileCheck %s

func.func @test_memref_store_scalar() {
  %ptr = memref.alloca() {test.ptr} : memref<f32>
  // CHECK: Successfully generated store for operation: %[[PTR:.*]] = memref.alloca() {test.ptr} : memref<f32>
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 4.200000e+01 : f32
  // CHECK: Generated: memref.store %[[VAL]], %[[PTR]][] : memref<f32>
  return
}

// -----

func.func @test_memref_store_int() {
  %ptr = memref.alloca() {test.ptr} : memref<i32>
  // CHECK: Successfully generated store for operation: %[[PTR:.*]] = memref.alloca() {test.ptr} : memref<i32>
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 42 : i32
  // CHECK: Generated: memref.store %[[VAL]], %[[PTR]][] : memref<i32>
  return
}

// -----

func.func @test_memref_store_i64() {
  %ptr = memref.alloca() {test.ptr} : memref<i64>
  // CHECK: Successfully generated store for operation: %[[PTR:.*]] = memref.alloca() {test.ptr} : memref<i64>
  // CHECK: Generated: %[[VAL:.*]] = arith.constant 42 : i64
  // CHECK: Generated: memref.store %[[VAL]], %[[PTR]][] : memref<i64>
  return
}

// -----

func.func @test_memref_store_dynamic() {
  %c10 = arith.constant 10 : index
  %ptr = memref.alloc(%c10) {test.ptr} : memref<?xf32>
  // CHECK: Failed to generate store for operation: %[[PTR:.*]] = memref.alloc(%{{.*}}) {test.ptr} : memref<?xf32>
  return
}


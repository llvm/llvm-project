// RUN: mlir-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(test-acc-pointer-like-interface{test-mode=free}))" 2>&1 | FileCheck %s

func.func @test_static_memref_free() {
  %0 = memref.alloca() {test.ptr} : memref<10x20xf32>
  // CHECK: Successfully generated free for operation: %[[ORIG:.*]] = memref.alloca() {test.ptr} : memref<10x20xf32>
  // CHECK-NOT: Generated
  return
}

// -----

func.func @test_dynamic_memref_free() {
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %orig = memref.alloc(%c10, %c20) {test.ptr} : memref<?x?xf32>
  
  // CHECK: Successfully generated free for operation: %[[ORIG:.*]] = memref.alloc(%[[C10:.*]], %[[C20:.*]]) {test.ptr} : memref<?x?xf32>
  // CHECK: Generated: memref.dealloc %[[ORIG]] : memref<?x?xf32>
  return
}

// -----

func.func @test_cast_walking_free() {
  %0 = memref.alloca() : memref<10x20xf32>
  %1 = memref.cast %0 {test.ptr} : memref<10x20xf32> to memref<?x?xf32>
  
  // CHECK: Successfully generated free for operation: %[[CAST:.*]] = memref.cast %[[ALLOCA:.*]] {test.ptr} : memref<10x20xf32> to memref<?x?xf32>
  // CHECK-NOT: Generated
  return
}

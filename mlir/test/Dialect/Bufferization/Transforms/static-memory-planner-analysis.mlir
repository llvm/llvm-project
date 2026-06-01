// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file | FileCheck %s

// -----

// Test 1: Simple sequential alloc/dealloc pairs
// CHECK-LABEL: func @simple_sequential
func.func @simple_sequential() {
  // CHECK: %[[ARENA:.*]] = memref.alloc() : memref<1536xf32>
  // CHECK-NEXT: %[[SUBVIEW0:.*]] = memref.subview %[[ARENA]][0] [1024] [1] : memref<1536xf32> to memref<1024xf32, strided<[1]>>
  // CHECK-NEXT: %[[SUBVIEW1:.*]] = memref.subview %[[ARENA]][1024] [512] [1] : memref<1536xf32> to memref<512xf32, strided<[1], offset: 1024>>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  return
}

// -----

// Test 2: Dynamic shape - should be skipped (no transformation)
// CHECK-LABEL: func @dynamic_shape_skipped
func.func @dynamic_shape_skipped(%n: index) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc(%n) : memref<?xf32>
  return
}

// -----

// Test 3: No dealloc - should be skipped
// CHECK-LABEL: func @no_dealloc_skipped
func.func @no_dealloc_skipped() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc() : memref<1024xf32>
  return
}

// -----

// Test 4: Dealloc in different block - should be skipped
// CHECK-LABEL: func @different_block_skipped
func.func @different_block_skipped(%cond: i1) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: scf.if
  // CHECK: memref.dealloc %[[ALLOC]]
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc() : memref<1024xf32>
  scf.if %cond {
    memref.dealloc %alloc : memref<1024xf32>
    scf.yield
  }
  return
}

// -----

// Test 5: Overlapping lifetimes (both eligible, sequential offsets)
// CHECK-LABEL: func @overlapping_lifetimes
func.func @overlapping_lifetimes() {
  // CHECK: %[[ARENA:.*]] = memref.alloc() : memref<1536xf32>
  // CHECK-NEXT: %[[SUBVIEW0:.*]] = memref.subview %[[ARENA]][0] [512] [1] : memref<1536xf32> to memref<512xf32, strided<[1]>>
  // CHECK-NEXT: %[[SUBVIEW1:.*]] = memref.subview %[[ARENA]][512] [1024] [1] : memref<1536xf32> to memref<1024xf32, strided<[1], offset: 512>>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() : memref<512xf32>
  %alloc1 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc1 : memref<1024xf32>
  memref.dealloc %alloc0 : memref<512xf32>
  return
}

// -----

// Test 6: Multiple allocations with sequential offsets
// CHECK-LABEL: func @multiple_sequential
func.func @multiple_sequential() {
  // CHECK: %[[ARENA:.*]] = memref.alloc() : memref<3584xf32>
  // CHECK-NEXT: %[[SUBVIEW0:.*]] = memref.subview %[[ARENA]][0] [1024] [1] : memref<3584xf32> to memref<1024xf32, strided<[1]>>
  // CHECK-NEXT: %[[SUBVIEW1:.*]] = memref.subview %[[ARENA]][1024] [512] [1] : memref<3584xf32> to memref<512xf32, strided<[1], offset: 1024>>
  // CHECK-NEXT: %[[SUBVIEW2:.*]] = memref.subview %[[ARENA]][1536] [2048] [1] : memref<3584xf32> to memref<2048xf32, strided<[1], offset: 1536>>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  %alloc2 = memref.alloc() : memref<2048xf32>
  memref.dealloc %alloc2 : memref<2048xf32>
  return
}

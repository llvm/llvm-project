// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file -verify-diagnostics

// -----

// Test 1: Simple sequential alloc/dealloc pairs
func.func @simple_sequential() {
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  return
}

// -----

// Test 2: Dynamic shape - should be skipped
func.func @dynamic_shape_skipped(%n: index) {
  // expected-remark @below {{static-memory-planner: skip: dynamic shape}}
  %alloc = memref.alloc(%n) : memref<?xf32>
  return
}

// -----

// Test 3: No dealloc - should be skipped
func.func @no_dealloc_skipped() {
  // expected-remark @below {{static-memory-planner: skip: no unique dealloc}}
  %alloc = memref.alloc() : memref<1024xf32>
  return
}

// -----

// Test 4: Dealloc in different block - should be skipped
func.func @different_block_skipped(%cond: i1) {
  // expected-remark @below {{static-memory-planner: skip: dealloc in different block}}
  %alloc = memref.alloc() : memref<1024xf32>
  scf.if %cond {
    memref.dealloc %alloc : memref<1024xf32>
    scf.yield
  }
  return
}

// -----

// Test 5: Overlapping lifetimes
func.func @overlapping_lifetimes() {
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc0 = memref.alloc() : memref<512xf32>
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc1 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc1 : memref<1024xf32>
  memref.dealloc %alloc0 : memref<512xf32>
  return
}

// -----

// Test 6: Multiple allocations with non-overlapping lifetimes
func.func @multiple_reusable() {
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  // expected-remark @below {{static-memory-planner: eligible}}
  %alloc2 = memref.alloc() : memref<2048xf32>
  memref.dealloc %alloc2 : memref<2048xf32>
  return
}

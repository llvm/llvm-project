// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file | FileCheck %s

// -----

// Test 1: Simple sequential alloc/dealloc pairs
// CHECK-LABEL: func @simple_sequential
func.func @simple_sequential() {
  // Arena is i8 buffer: 1024*4 + 512*4 = 6144 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<6144xi8>
  // First allocation at offset 0
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<6144xi8> to memref<1024xf32>
  // Second allocation at offset 4096 bytes (1024 * 4)
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C4096]]][] : memref<6144xi8> to memref<512xf32>
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
  // Arena: 512*4 + 1024*4 = 6144 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<6144xi8>
  // First allocation at offset 0
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<6144xi8> to memref<512xf32>
  // Second allocation at offset 2048 bytes (512 * 4)
  // CHECK-NEXT: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C2048]]][] : memref<6144xi8> to memref<1024xf32>
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
  // Arena: 1024*4 + 512*4 + 2048*4 = 14336 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<14336xi8>
  // First at offset 0
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<14336xi8> to memref<1024xf32>
  // Second at offset 4096 bytes (1024 * 4)
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C4096]]][] : memref<14336xi8> to memref<512xf32>
  // Third at offset 6144 bytes (1024*4 + 512*4)
  // CHECK-NEXT: %[[C6144:.*]] = arith.constant 6144 : index
  // CHECK-NEXT: %[[VIEW2:.*]] = memref.view %[[ARENA]][%[[C6144]]][] : memref<14336xi8> to memref<2048xf32>
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

// -----

// Test 7: Alignment requirements with padding
// CHECK-LABEL: func @alignment_padding
func.func @alignment_padding() {
  // Arena has max alignment (128 bytes), total: 256*4 + 128*4 + 64*4 = 1792 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1792xi8>
  // First alloc: 256 f32, alignment=128, offset=0 bytes (128-aligned)
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<1792xi8> to memref<256xf32>
  // Second alloc: 128 f32, alignment=64, offset=1024 bytes (64-aligned)
  // CHECK-NEXT: %[[C1024:.*]] = arith.constant 1024 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C1024]]][] : memref<1792xi8> to memref<128xf32>
  // Third alloc: 64 f32, alignment=128, offset=1536 bytes (128-aligned)
  // CHECK-NEXT: %[[C1536:.*]] = arith.constant 1536 : index
  // CHECK-NEXT: %[[VIEW2:.*]] = memref.view %[[ARENA]][%[[C1536]]][] : memref<1792xi8> to memref<64xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() {alignment = 128 : i64} : memref<256xf32>
  memref.dealloc %alloc0 : memref<256xf32>
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  memref.dealloc %alloc1 : memref<128xf32>
  %alloc2 = memref.alloc() {alignment = 128 : i64} : memref<64xf32>
  memref.dealloc %alloc2 : memref<64xf32>
  return
}

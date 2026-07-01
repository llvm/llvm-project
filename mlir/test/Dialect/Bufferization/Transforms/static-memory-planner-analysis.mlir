// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file | FileCheck %s

// -----

// Test 1: Sequential alloc and dealloc pairs.
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

// Test 2: Non-sequential pairs (alloc alloc dealloc dealloc).
// CHECK-LABEL: func @non_sequential_pairs
func.func @non_sequential_pairs() {
  // Arena: 1024*4 + 512*4 = 6144 bytes
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
  %alloc1 = memref.alloc() : memref<512xf32>
  memref.dealloc %alloc0 : memref<1024xf32>
  memref.dealloc %alloc1 : memref<512xf32>
  return
}

// -----

// Test 3: Interleaved pairs (alloc alloc dealloc alloc dealloc dealloc).
// CHECK-LABEL: func @interleaved_pairs
func.func @interleaved_pairs() {
  // Arena: 512*4 + 256*4 + 128*4 = 3584 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<3584xi8>
  // First at offset 0
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<3584xi8> to memref<512xf32>
  // Second at offset 2048 bytes (512 * 4)
  // CHECK-NEXT: %[[C2048:.*]] = arith.constant 2048 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C2048]]][] : memref<3584xi8> to memref<256xf32>
  // Third at offset 3072 bytes (512*4 + 256*4)
  // CHECK-NEXT: %[[C3072:.*]] = arith.constant 3072 : index
  // CHECK-NEXT: %[[VIEW2:.*]] = memref.view %[[ARENA]][%[[C3072]]][] : memref<3584xi8> to memref<128xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() : memref<512xf32>
  %alloc1 = memref.alloc() : memref<256xf32>
  memref.dealloc %alloc0 : memref<512xf32>
  %alloc2 = memref.alloc() : memref<128xf32>
  memref.dealloc %alloc1 : memref<256xf32>
  memref.dealloc %alloc2 : memref<128xf32>
  return
}

// -----

// Test 4: Dynamic shape - should be skipped (no transformation)
// CHECK-LABEL: func @dynamic_shape_skipped
func.func @dynamic_shape_skipped(%n: index) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc(%n) : memref<?xf32>
  return
}

// -----

// Test 5: No dealloc - should be skipped
// CHECK-LABEL: func @no_dealloc_skipped
func.func @no_dealloc_skipped() {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.subview
  %alloc = memref.alloc() : memref<1024xf32>
  return
}

// -----

// Test 6: Dealloc in different block - should be skipped
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

// Test 7: Multiple allocations with sequential offsets
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

// Test 8: Alignment requirements with padding
// CHECK-LABEL: func @alignment_padding
func.func @alignment_padding() {
  // Arena: 256*4 + 128*4 + 64*4 = 1792 bytes, alignment = lcm(128,64,128) = 128
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

// -----

// Test 9: LCM arena alignment (alignment=4, alignment=16 → lcm=16).
// For power-of-2 alignments lcm equals max, but lcm is the correct
// general formula. Arena must be aligned to 16 so that all views are
// correctly aligned regardless of their individual requirements.
// CHECK-LABEL: func @lcm_alignment
func.func @lcm_alignment() {
  // Arena: 3*4 + 3*4 = 24 bytes, but second alloc needs 16-byte offset
  // (alignTo(12, 16) = 16), so total = 28 bytes, alignment = lcm(4,16) = 16
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 16 : i64} : memref<28xi8>
  // First at offset 0 (alignment=4, 0 % 4 == 0)
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<28xi8> to memref<3xi32>
  // Second at offset 16 (alignment=16, 16 % 16 == 0)
  // CHECK-NEXT: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C16]]][] : memref<28xi8> to memref<3xi32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() {alignment = 4 : i64} : memref<3xi32>
  memref.dealloc %alloc0 : memref<3xi32>
  %alloc1 = memref.alloc() {alignment = 16 : i64} : memref<3xi32>
  memref.dealloc %alloc1 : memref<3xi32>
  return
}

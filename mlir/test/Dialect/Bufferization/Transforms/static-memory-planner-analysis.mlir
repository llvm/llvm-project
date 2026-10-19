// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis))" \
// RUN:     -split-input-file | FileCheck %s

// -----

// Test 1: Sequential alloc and dealloc pairs.
// CHECK-LABEL: func @simple_sequential
func.func @simple_sequential() {
  // Arena is i8 buffer: 1024*4 + 512*4 = 6144 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<6144xi8>
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
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<6144xi8>
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
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<3584xi8>
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

// Test 5: Multiple allocations with sequential offsets
// CHECK-LABEL: func @multiple_sequential
func.func @multiple_sequential() {
  // Arena: 1024*4 + 512*4 + 2048*4 = 14336 bytes
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<14336xi8>
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

// Test 6: Alignment requirements with padding
// CHECK-LABEL: func @alignment_padding
func.func @alignment_padding() {
  // Arena: 256*4 + 128*4 + 64*4 = 1792 bytes, alignment = lcm(128,64,128) = 128
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 128 : memref<1792xi8>
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
  %alloc0 = memref.alloc() alignment = 128 : memref<256xf32>
  memref.dealloc %alloc0 : memref<256xf32>
  %alloc1 = memref.alloc() alignment = 64 : memref<128xf32>
  memref.dealloc %alloc1 : memref<128xf32>
  %alloc2 = memref.alloc() alignment = 128 : memref<64xf32>
  memref.dealloc %alloc2 : memref<64xf32>
  return
}

// -----

// Test 7: LCM arena alignment (alignment=4, alignment=16 → lcm=16).
// For power-of-2 alignments lcm equals max, but lcm is the correct
// general formula. Arena must be aligned to 16 so that all views are
// correctly aligned regardless of their individual requirements.
// CHECK-LABEL: func @lcm_alignment
func.func @lcm_alignment() {
  // Arena: 3*4 + 3*4 = 24 bytes, but second alloc needs 16-byte offset
  // (alignTo(12, 16) = 16), so total = 28 bytes, alignment = lcm(4,16) = 16
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 16 : memref<28xi8>
  // First at offset 0 (alignment=4, 0 % 4 == 0)
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VIEW0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<28xi8> to memref<3xi32>
  // Second at offset 16 (alignment=16, 16 % 16 == 0)
  // CHECK-NEXT: %[[C16:.*]] = arith.constant 16 : index
  // CHECK-NEXT: %[[VIEW1:.*]] = memref.view %[[ARENA]][%[[C16]]][] : memref<28xi8> to memref<3xi32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc0 = memref.alloc() alignment = 4 : memref<3xi32>
  memref.dealloc %alloc0 : memref<3xi32>
  %alloc1 = memref.alloc() alignment = 16 : memref<3xi32>
  memref.dealloc %alloc1 : memref<3xi32>
  return
}

// -----

// Test 8: Single alloc freed via arith.select-based dealloc.
// CHECK-LABEL: func @select_single_alloc
func.func @select_single_alloc() {
  %c = arith.constant true
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<4096xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[V:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<4096xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %alloc = memref.alloc() : memref<1024xf32>
  %sel = arith.select %c, %alloc, %alloc : memref<1024xf32>
  memref.dealloc %sel : memref<1024xf32>
  return
}

// -----

// Test 9: Two allocs freed via a shared select-based dealloc.
// Group constraint: both must be eligible together or neither is.
// CHECK-LABEL: func @select_shared_dealloc
func.func @select_shared_dealloc() {
  %c = arith.constant true
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[V0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %[[V1:.*]] = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %sel = arith.select %c, %a, %b : memref<1024xf32>
  memref.dealloc %sel : memref<1024xf32>
  return
}

// -----

// Test 10: Two allocs, two select-based deallocs (mentor's canonical example).
// %a freed via dealloc(%sel1) or dealloc(%sel2), %b likewise.
// CHECK-LABEL: func @select_two_deallocs
func.func @select_two_deallocs() {
  %c = arith.constant true
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %sel1 = arith.select %c, %a, %b : memref<1024xf32>
  memref.dealloc %sel1 : memref<1024xf32>
  %sel2 = arith.select %c, %b, %a : memref<1024xf32>
  memref.dealloc %sel2 : memref<1024xf32>
  return
}

// -----

// Test 11: Deallocs nested inside scf.if bodies (mentor case_2).
// Both allocs live in the entry block; each dealloc is anchored by the
// enclosing scf.if, so both are eligible via the buffer view-flow analysis.
// CHECK-LABEL: func @scf_if_nested_deallocs
func.func @scf_if_nested_deallocs(%c: i1, %d: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  scf.if %c {
    memref.dealloc %a : memref<1024xf32>
  }
  scf.if %d {
    memref.dealloc %b : memref<1024xf32>
  }
  return
}

// -----

// Test 12: Allocs flow through scf.if results, then deallocated (mentor case_1).
// The analysis follows the scf.if result aliases back to %a and %b, so both
// are planned and the yielded views are rewired automatically.
// CHECK-LABEL: func @scf_if_result_aliases
func.func @scf_if_result_aliases(%c: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[V0:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %[[V1:.*]] = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  // CHECK: scf.if
  // CHECK: scf.yield %[[V0]]
  // CHECK: scf.yield %[[V1]]
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %0 = scf.if %c -> memref<1024xf32> {
    scf.yield %a : memref<1024xf32>
  } else {
    scf.yield %b : memref<1024xf32>
  }
  %1 = scf.if %c -> memref<1024xf32> {
    scf.yield %b : memref<1024xf32>
  } else {
    scf.yield %a : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  memref.dealloc %1 : memref<1024xf32>
  return
}

// -----

// Test 13: Alloc nested inside a conditional/loop body is left untouched.
// Only entry-block allocs are planned; the nested %b keeps its alloc/dealloc.
// CHECK-LABEL: func @scf_if_nested_alloc_skipped
func.func @scf_if_nested_alloc_skipped(%c: i1) {
  // CHECK-NOT: memref.view
  // CHECK: scf.if
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: memref.dealloc
  scf.if %c {
    %b = memref.alloc() : memref<1024xf32>
    memref.dealloc %b : memref<1024xf32>
  }
  return
}

// -----

// Test 14: A dealloc that may free both an entry-block alloc and a nested
// alloc (mentor case_3) is conservatively skipped: erasing it would be unsafe
// for the buffer that is not managed by the arena.
// CHECK-LABEL: func @scf_if_shared_nested_dealloc_skipped
func.func @scf_if_shared_nested_dealloc_skipped(%c: i1) {
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.view
  // CHECK: scf.if
  // CHECK: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %0 = scf.if %c -> memref<1024xf32> {
    memref.dealloc %a : memref<1024xf32>
    %b = memref.alloc() : memref<1024xf32>
    scf.yield %b : memref<1024xf32>
  } else {
    scf.yield %a : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  return
}
// -----

// Test 15: Alloc nested inside an scf.for body is left untouched (same rule as
// test 13 — only entry-block allocs are planned).
// CHECK-LABEL: func @scf_for_nested_alloc_skipped
func.func @scf_for_nested_alloc_skipped(%lb: index, %ub: index, %step: index) {
  // CHECK-NOT: memref.view
  // CHECK: scf.for
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: memref.dealloc
  scf.for %iv = %lb to %ub step %step {
    %b = memref.alloc() : memref<1024xf32>
    memref.dealloc %b : memref<1024xf32>
  }
  return
}

// -----

// Test 16: Entry-block alloc passed as scf.for iter_arg; each iteration frees
// the current iter_arg and allocates a fresh buffer. The reverse-alias guard
// conservatively skips %a because dealloc(%arg0) may also free the per-iteration
// nested %b (which is not managed by the arena).
// CHECK-LABEL: func @scf_for_iter_arg_nested_alloc
func.func @scf_for_iter_arg_nested_alloc(%lb: index, %ub: index, %step: index) {
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.view
  // CHECK: scf.for
  // CHECK: memref.dealloc
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %a) -> memref<1024xf32> {
    memref.dealloc %arg0 : memref<1024xf32>
    %b = memref.alloc() : memref<1024xf32>
    scf.yield %b : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  return
}

// -----

// Test 17: Entry-block alloc passed as scf.for iter_arg; each iteration
// allocates a fresh buffer and yields it without freeing the previous iter_arg
// (potential memory leak at runtime if the loop executes). The reverse-alias
// guard skips %a because dealloc(%0) may also free the nested per-iteration %b.
// CHECK-LABEL: func @scf_for_nested_alloc_yielded
func.func @scf_for_nested_alloc_yielded(%lb: index, %ub: index, %step: index) {
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.view
  // CHECK: scf.for
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %a) -> memref<1024xf32> {
    %b = memref.alloc() : memref<1024xf32>
    scf.yield %b : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  return
}

// -----

// Test 18: Entry-block alloc passed as scf.for iter_arg; the original %a is
// freed directly inside the loop body (not via the iter_arg), and a fresh
// buffer is allocated and yielded (potential double-free / memory leak at
// runtime if the loop executes more than once). The reverse-alias guard skips
// %a because dealloc(%0) may also free the nested per-iteration %b.
// CHECK-LABEL: func @scf_for_orig_alloc_freed_in_body
func.func @scf_for_orig_alloc_freed_in_body(%lb: index, %ub: index, %step: index) {
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK-NOT: memref.view
  // CHECK: scf.for
  // CHECK: memref.dealloc
  // CHECK: memref.alloc() : memref<1024xf32>
  // CHECK: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %0 = scf.for %iv = %lb to %ub step %step iter_args(%arg0 = %a) -> memref<1024xf32> {
    memref.dealloc %a : memref<1024xf32>
    %b = memref.alloc() : memref<1024xf32>
    scf.yield %b : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  return
}
// -----

// Test 19: Mixed static and dynamic shapes in the same function. The static
// alloc is transformed into the arena; the dynamic one is silently skipped and
// left as-is. The two kinds coexist safely in the same function.
// CHECK-LABEL: func @mixed_static_dynamic
func.func @mixed_static_dynamic(%n: index) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<4096xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<4096xi8> to memref<1024xf32>
  // CHECK: memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc(%n) : memref<?xf32>
  memref.dealloc %a : memref<1024xf32>
  return
}

// -----

// Test 20: Both branches of an scf.if dealloc the same alloc. The analysis
// finds both dealloc ops; the lifetime anchors at the scf.if, and the alloc is
// placed in the arena with both deallocs erased.
// CHECK-LABEL: func @scf_if_both_branches_dealloc
func.func @scf_if_both_branches_dealloc(%c: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<4096xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<4096xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  scf.if %c {
    memref.dealloc %a : memref<1024xf32>
  } else {
    memref.dealloc %a : memref<1024xf32>
  }
  return
}

// -----

// Test 21: Dealloc at depth 3 (scf.if inside scf.if inside scf.if).
// findAncestorOpInBlock returns the outermost scf.if as the anchor, making the
// lifetime conservative. The alloc still transforms correctly.
// CHECK-LABEL: func @deep_nested_dealloc
func.func @deep_nested_dealloc(%c1: i1, %c2: i1, %c3: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<4096xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<4096xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  scf.if %c1 {
    scf.if %c2 {
      scf.if %c3 {
        memref.dealloc %a : memref<1024xf32>
      }
    }
  }
  return
}

// -----

// Test 22: Two scf.if ops chained through their results. The alias chain is
// %a/%b → %0 → %1 → dealloc. The analysis resolves the full multi-hop chain,
// finds the single dealloc on %1, and transforms both allocs into the arena.
// CHECK-LABEL: func @chained_scf_if_results
func.func @chained_scf_if_results(%c1: i1, %c2: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %0 = scf.if %c1 -> memref<1024xf32> {
    scf.yield %a : memref<1024xf32>
  } else {
    scf.yield %b : memref<1024xf32>
  }
  %1 = scf.if %c2 -> memref<1024xf32> {
    scf.yield %0 : memref<1024xf32>
  } else {
    scf.yield %a : memref<1024xf32>
  }
  memref.dealloc %1 : memref<1024xf32>
  return
}

// -----

// Test 23: arith.select feeds into an scf.if result which is then deallocated.
// This is a cross-op-type alias chain: %a/%b → arith.select → scf.if → dealloc.
// Both ops implement different interfaces (BufferViewFlowOpInterface and
// RegionBranchOpInterface), so this exercises the unified analysis path.
// CHECK-LABEL: func @select_chained_into_scf_if
func.func @select_chained_into_scf_if(%c1: i1, %c2: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  %sel = arith.select %c1, %a, %b : memref<1024xf32>
  %0 = scf.if %c2 -> memref<1024xf32> {
    scf.yield %sel : memref<1024xf32>
  } else {
    scf.yield %a : memref<1024xf32>
  }
  memref.dealloc %0 : memref<1024xf32>
  return
}

// -----

// Test 24: Mixed dealloc locations — one alloc freed inside an scf.if body,
// another freed directly in the entry block. Both live in the entry block, so
// both are eligible. They share the same arena despite different dealloc styles.
// CHECK-LABEL: func @mixed_dealloc_locations
func.func @mixed_dealloc_locations(%c: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<6144xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<6144xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<6144xi8> to memref<512xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<512xf32>
  scf.if %c {
    memref.dealloc %a : memref<1024xf32>
  }
  memref.dealloc %b : memref<512xf32>
  return
}

// -----

// Test 25: Dealloc nested in the else branch of a nested scf.if. Verifies
// that findAncestorOpInBlock works for else regions as well as then regions,
// and that the alias analysis traverses both sides of conditionals.
// CHECK-LABEL: func @nested_else_dealloc
func.func @nested_else_dealloc(%c1: i1, %c2: i1) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<4096xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<4096xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %a = memref.alloc() : memref<1024xf32>
  scf.if %c1 {
  } else {
    scf.if %c2 {
      memref.dealloc %a : memref<1024xf32>
    }
  }
  return
}

// -----

// Test 26: scf.for body only reads entry-block buffers (no ownership transfer,
// no iter_args). Both allocs and deallocs are in the entry block, so the
// transformation applies cleanly and the loop body receives the arena views.
// CHECK-LABEL: func @scf_for_reads_entry_block_bufs
func.func @scf_for_reads_entry_block_bufs(%lb: index, %ub: index, %step: index) {
  // CHECK: %[[ARENA:.*]] = memref.alloc() alignment = 1 : memref<8192xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[VA:.*]] = memref.view %[[ARENA]][%[[C0]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %[[VB:.*]] = memref.view %[[ARENA]][%[[C4096]]][] : memref<8192xi8> to memref<1024xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  // CHECK: scf.for
  // CHECK: memref.copy %[[VA]], %[[VB]]
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<1024xf32>
  scf.for %iv = %lb to %ub step %step {
    memref.copy %a, %b : memref<1024xf32> to memref<1024xf32>
  }
  memref.dealloc %a : memref<1024xf32>
  memref.dealloc %b : memref<1024xf32>
  return
}

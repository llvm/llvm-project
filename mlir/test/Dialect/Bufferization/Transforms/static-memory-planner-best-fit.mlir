// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(static-memory-planner-analysis{algorithm=best-fit}))" \
// RUN:     -split-input-file | FileCheck %s

// -----

// Test 1: Non-overlapping lifetimes reuse the same memory.
// With trivial packing this would be 8192 bytes; best-fit reuses the space.
// CHECK-LABEL: func @reuse_non_overlapping
func.func @reuse_non_overlapping() {
  // Arena should be 4096 bytes (1024 * 4), not 8192.
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<4096xi8>
  // First allocation at offset 0
  // CHECK-NEXT: %[[C0_0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0_0]]][] : memref<4096xi8> to memref<1024xf32>
  // Second allocation also at offset 0 (reuses freed space)
  // CHECK-NEXT: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0_1]]][] : memref<4096xi8> to memref<1024xf32>
  %0 = memref.alloc() : memref<1024xf32>
  memref.dealloc %0 : memref<1024xf32>
  %1 = memref.alloc() : memref<1024xf32>
  memref.dealloc %1 : memref<1024xf32>
  return
}

// -----

// Test 2: Overlapping lifetimes cannot reuse memory.
// CHECK-LABEL: func @no_reuse_overlapping
func.func @no_reuse_overlapping() {
  // Both are live at the same time, so arena = 4096 + 2048 = 6144 bytes.
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<6144xi8>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C0]]][] : memref<6144xi8> to memref<1024xf32>
  // CHECK-NEXT: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]][%[[C4096]]][] : memref<6144xi8> to memref<512xf32>
  %0 = memref.alloc() : memref<1024xf32>
  %1 = memref.alloc() : memref<512xf32>
  memref.dealloc %0 : memref<1024xf32>
  memref.dealloc %1 : memref<512xf32>
  return
}

// -----

// Test 3: Best-fit picks the smallest suitable gap.
// Layout: A(4096) at 0, B(1024) at 4096, C(4096) at 5120, D(1024) at 9216.
// B and D are freed while A and C are still live, creating two gaps:
//   [4096, 5120) = 1024 bytes (B's slot)
//   [9216, 10240) = 1024 bytes (D's slot)
// Then we free A, creating gap [0, 4096) = 4096 bytes.
// Now allocate E(512 bytes). Gaps: [0,4096)=4096, [4096,5120)=1024, [9216,10240)=1024.
// Best-fit should pick one of the 1024-byte gaps (smallest fit for 512).
// CHECK-LABEL: func @best_fit_smallest_gap
func.func @best_fit_smallest_gap() {
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 1 : i64} : memref<10240xi8>
  // A at offset 0
  // CHECK-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]
  // B at offset 4096
  // CHECK-NEXT: %{{.*}} = arith.constant 4096 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]
  // C at offset 5120
  // CHECK-NEXT: %{{.*}} = arith.constant 5120 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]
  // D at offset 9216
  // CHECK-NEXT: %{{.*}} = arith.constant 9216 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]
  // E at offset 9216 (best-fit picks 1024-byte trailing gap over 5120-byte gap)
  // CHECK-NEXT: %{{.*}} = arith.constant 9216 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]
  %a = memref.alloc() : memref<1024xf32>
  %b = memref.alloc() : memref<256xf32>
  %c = memref.alloc() : memref<1024xf32>
  %d = memref.alloc() : memref<256xf32>
  memref.dealloc %b : memref<256xf32>
  memref.dealloc %d : memref<256xf32>
  memref.dealloc %a : memref<1024xf32>
  %e = memref.alloc() : memref<128xf32>
  memref.dealloc %c : memref<1024xf32>
  memref.dealloc %e : memref<128xf32>
  return
}

// -----

// Test 4: Alignment padding can disqualify a gap.
// A(128, align 128) at 0 (live), B(56, align 1) at 128, C(128, align 128) at 256 (live).
// B is freed => gap [128, 256) = 128 bytes.
// D(64, align 128): aligned start = 128, fits in gap.
// E(64, align 256): next 256-aligned offset in [128,256) is 256 = gap end, doesn't fit.
//   E must go past the arena high-water mark.
// CHECK-LABEL: func @best_fit_alignment
func.func @best_fit_alignment() {
  // CHECK: %[[ARENA:.*]] = memref.alloc() {alignment = 256 : i64} : memref<576xi8>
  // CHECK-NEXT: %{{.*}} = arith.constant 0 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]{{.*}} to memref<128xi8>
  // CHECK-NEXT: %{{.*}} = arith.constant 128 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]{{.*}} to memref<56xi8>
  // CHECK-NEXT: %{{.*}} = arith.constant 256 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]{{.*}} to memref<128xi8>
  // D fits in gap at offset 128
  // CHECK-NEXT: %{{.*}} = arith.constant 128 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]{{.*}} to memref<64xi8>
  // E cannot fit in gap (alignment 256), placed at offset 512
  // CHECK-NEXT: %{{.*}} = arith.constant 512 : index
  // CHECK-NEXT: %{{.*}} = memref.view %[[ARENA]]{{.*}} to memref<64xi8>
  %a = memref.alloc() {alignment = 128 : i64} : memref<128xi8>
  %b = memref.alloc() : memref<56xi8>
  %c = memref.alloc() {alignment = 128 : i64} : memref<128xi8>
  memref.dealloc %b : memref<56xi8>
  %d = memref.alloc() {alignment = 128 : i64} : memref<64xi8>
  %e = memref.alloc() {alignment = 256 : i64} : memref<64xi8>
  memref.dealloc %a : memref<128xi8>
  memref.dealloc %c : memref<128xi8>
  memref.dealloc %d : memref<64xi8>
  memref.dealloc %e : memref<64xi8>
  return
}

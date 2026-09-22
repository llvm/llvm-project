; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

; CGP's splitLargeGEPOffsets now handles negative offsets by rebasing to the
; minimum offset directly (rather than a 4096-byte-aligned high part), so the
; residual falls within LDP/STP's pairing range. This mirrors what
; shareBaseAddresses does at MIR level, but earlier — at IR level — so ISel
; never materializes independent SUBXri bases and LSO can pair directly.

define void @neg_small_i32(ptr %p) {
; CHECK-LABEL: neg_small_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #400
; CHECK-NEXT:    ldp w9, w8, [x8]
; CHECK-NEXT:    str w9, [x8]
; CHECK-NEXT:    str w8, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i32, ptr %p, i64 -100
  %gep1 = getelementptr i32, ptr %p, i64 -99
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

define void @neg_small_i64(ptr %p) {
; CHECK-LABEL: neg_small_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #800
; CHECK-NEXT:    ldp x9, x8, [x8]
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    str x8, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i64, ptr %p, i64 -100
  %gep1 = getelementptr i64, ptr %p, i64 -99
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

define void @neg_4k_aligned_i64(ptr %p) {
; 4096-byte-aligned negative offset: single SUBXri with shift=12.
; CHECK-LABEL: neg_4k_aligned_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #2, lsl #12 // =8192
; CHECK-NEXT:    ldp x9, x8, [x8]
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    str x8, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i64, ptr %p, i64 -1024
  %gep1 = getelementptr i64, ptr %p, i64 -1023
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

define void @neg_large_i64(ptr %p) {
; Large negative offset not aligned to 4096: ISel splits into two SUBXri,
; but CGP still establishes a shared base so LSO can pair.
; CHECK-LABEL: neg_large_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #19, lsl #12 // =77824
; CHECK-NEXT:    sub x8, x8, #2176
; CHECK-NEXT:    ldp x9, x8, [x8]
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    str x8, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i64, ptr %p, i64 -10000
  %gep1 = getelementptr i64, ptr %p, i64 -9999
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

define void @neg_3sibling_i32(ptr %p) {
; CHECK-LABEL: neg_3sibling_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #400
; CHECK-NEXT:    ldp w9, w10, [x8]
; CHECK-NEXT:    ldr w8, [x8, #8]
; CHECK-NEXT:    str w9, [x8]
; CHECK-NEXT:    str w10, [x8]
; CHECK-NEXT:    str w8, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i32, ptr %p, i64 -100
  %gep1 = getelementptr i32, ptr %p, i64 -99
  %gep2 = getelementptr i32, ptr %p, i64 -98
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  %v2 = load i32, ptr %gep2
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  store volatile i32 %v2, ptr poison
  ret void
}

define void @pos_small_i64(ptr %p) {
; Positive offset: behavior unchanged (no CGP rebase needed, LDR encodes directly).
; CHECK-LABEL: pos_small_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr x8, [x0, #800]
; CHECK-NEXT:    ldr x9, [x0, #808]
; CHECK-NEXT:    str x8, [x8]
; CHECK-NEXT:    str x9, [x8]
; CHECK-NEXT:    ret
  %gep0 = getelementptr i64, ptr %p, i64 100
  %gep1 = getelementptr i64, ptr %p, i64 101
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

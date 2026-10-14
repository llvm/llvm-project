; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

; CGP's splitLargeGEPOffsets handles negative offsets the same way as positive
; ones: rebasing to a 4096-byte-aligned high part (HighPart = MinOffset & ~0xfff)
; so each residual fits LDR/STR's 12-bit unsigned scaled immediate. The base is
; materialized as a single SUBXri (for negative HighPart) or ADDXri (for
; positive). With -aarch64-ldp-stp-base-adjust (default on), LSO further folds
; the SUBXri/ADDXri with the residual and pairs the loads/stores into LDP/STP
; when the combined offset is encodable.

define void @neg_small_i32(ptr %p) {
; HighPart = -4096; residuals 3696, 3700. LSO folds -4096+3696 = -400 and pairs.
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
; HighPart = -4096; residuals 3296, 3304. LSO folds -4096+3296 = -800 and pairs.
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
; 4096-byte-aligned negative offset: HighPart = MinOffset, residuals 0, 8 fit
; LDP's 7-bit range, so LSO pairs directly.
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
; Large negative offset: HighPart = -81920 (single SUBXri shift=12). Residuals
; 1920, 1928 exceed LDP's 7-bit range but LSO inserts a separate ADDXri and pairs.
; CHECK-LABEL: neg_large_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #20, lsl #12 // =81920
; CHECK-NEXT:    add x9, x8, #1920
; CHECK-NEXT:    ldp x9, x8, [x9]
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
; HighPart = -4096; residuals 3696, 3700, 3704. First two pair via LSO fallback
; (ADDXri + LDP); third stays independent (Reg1ClobbersBase: w8 aliases base x8).
; CHECK-LABEL: neg_3sibling_i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    sub x8, x0, #1, lsl #12 // =4096
; CHECK-NEXT:    add x0, x8, #3696
; CHECK-NEXT:    ldr w8, [x8, #3704]
; CHECK-NEXT:    ldp w9, w10, [x0]
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
; Positive offset: LSO inserts ADDXri and pairs (no CGP rebase needed).
; CHECK-LABEL: pos_small_i64:
; CHECK:       // %bb.0:
; CHECK-NEXT:    add x8, x0, #800
; CHECK-NEXT:    ldp x8, x9, [x8]
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

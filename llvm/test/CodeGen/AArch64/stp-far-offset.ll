; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,ENABLED
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -aarch64-ldp-stp-base-adjust=0 < %s | FileCheck %s --check-prefixes=CHECK,DISABLED

define void @stp_far_offset_i32(ptr %p, i32 %v0, i32 %v1) {
; ENABLED-LABEL: stp_far_offset_i32:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #400
; ENABLED-NEXT:    stp w1, w2, [x[[TMP]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_i32:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     stp w
; DISABLED-NOT:     stp x
; DISABLED:        ret
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  store i32 %v0, ptr %gep0
  store i32 %v1, ptr %gep1
  ret void
}

define void @stp_far_offset_i64(ptr %p, i64 %v0, i64 %v1) {
; ENABLED-LABEL: stp_far_offset_i64:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #800
; ENABLED-NEXT:    stp x1, x2, [x[[TMP]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_i64:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     stp w
; DISABLED-NOT:     stp x
; DISABLED:        ret
  %gep0 = getelementptr i64, ptr %p, i64 100
  %gep1 = getelementptr i64, ptr %p, i64 101
  store i64 %v0, ptr %gep0
  store i64 %v1, ptr %gep1
  ret void
}

; Regression guard for the base-adjust scratch scavenger when the store
; value registers are sub-64-bit (Wn): the scratch candidates are 64-bit
; (X9..X15), so an exact register compare would miss that X9 aliases W9
; and pick a scratch that clobbers the stored value before the STP reads
; it.  The inline-asm constraints pin the stored values to W9 and W8, and
; the scratch capture `1[0-5]` cannot match 8 or 9, so this fails if the
; aliasing-aware skip is ever dropped.
define void @stp_far_offset_i32_w9w8_dst(ptr %p) {
; ENABLED-LABEL: stp_far_offset_i32_w9w8_dst:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[SCRATCH:1[0-5]]], x0, #400
; ENABLED:         stp w9, w8, [x[[SCRATCH]]]
; ENABLED:         ret
;
; DISABLED-LABEL: stp_far_offset_i32_w9w8_dst:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     stp w
; DISABLED-NOT:     stp x
; DISABLED:        ret
  %v0 = call i32 asm "", "={w9}"()
  %v1 = call i32 asm "", "={w8}"()
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  store i32 %v0, ptr %gep0
  store i32 %v1, ptr %gep1
  ret void
}

define void @stp_near_offset(ptr %p, i32 %v0, i32 %v1) {
; Near offsets within STP range always pair regardless of the option.
; CHECK-LABEL: stp_near_offset:
; CHECK:       // %bb.0:
; CHECK-NOT:     add
; CHECK:        stp w{{[0-9]+}}, w{{[0-9]+}}, [x0, #40]
; CHECK:        ret
  %gep0 = getelementptr i32, ptr %p, i64 10
  %gep1 = getelementptr i32, ptr %p, i64 11
  store i32 %v0, ptr %gep0
  store i32 %v1, ptr %gep1
  ret void
}

define void @stp_far_offset_interleaved(ptr %p, i32 %v0, i32 %v1) {
; Intervening non-aliasing operation between two far-offset stores.
; ENABLED-LABEL: stp_far_offset_interleaved:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #400
; ENABLED:       stp w{{[0-9]+}}, w{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: stp_far_offset_interleaved:
; DISABLED-NOT:   stp w
; DISABLED-NOT:   stp x
; DISABLED:       ret
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  %stack = alloca i32
  store i32 %v0, ptr %gep0
  %v2 = load volatile i32, ptr %stack
  store i32 %v1, ptr %gep1
  ret void
}

; Offset = 1024 * 4 = 4096 bytes = 0x1000
; The ADDXri encodes as #1, LSL #12 which equals 4096.
define void @stp_far_offset_4k_aligned_i32(ptr %p, i32 %v0, i32 %v1) {
; ENABLED-LABEL: stp_far_offset_4k_aligned_i32:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #1, lsl #12
; ENABLED:       stp w{{[0-9]+}}, w{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: stp_far_offset_4k_aligned_i32:
; DISABLED-NOT:   stp w
; DISABLED-NOT:   stp x
; DISABLED:       ret
  %gep0 = getelementptr i32, ptr %p, i64 1024
  %gep1 = getelementptr i32, ptr %p, i64 1025
  store i32 %v0, ptr %gep0
  store i32 %v1, ptr %gep1
  ret void
}

; Offset = 500 * 8 = 4000 bytes (fits in shift=0, <= 4095)
define void @stp_far_offset_large_i64(ptr %p, i64 %v0, i64 %v1) {
; ENABLED-LABEL: stp_far_offset_large_i64:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #4000
; ENABLED:       stp x{{[0-9]+}}, x{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: stp_far_offset_large_i64:
; DISABLED-NOT:   stp w
; DISABLED-NOT:   stp x
; DISABLED:       ret
  %gep0 = getelementptr i64, ptr %p, i64 500
  %gep1 = getelementptr i64, ptr %p, i64 501
  store i64 %v0, ptr %gep0
  store i64 %v1, ptr %gep1
  ret void
}

define void @stp_far_offset_q(ptr %p, <2 x i64> %v0, <2 x i64> %v1) {
; ENABLED-LABEL: stp_far_offset_q:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #1600
; ENABLED-NEXT:    stp q0, q1, [x[[TMP]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_q:
; DISABLED:       // %bb.0:
; DISABLED-NEXT:    str q0, [x0, #1600]
; DISABLED-NEXT:    str q1, [x0, #1616]
; DISABLED-NEXT:    ret
  %gep0 = getelementptr <2 x i64>, ptr %p, i64 100
  %gep1 = getelementptr <2 x i64>, ptr %p, i64 101
  store <2 x i64> %v0, ptr %gep0
  store <2 x i64> %v1, ptr %gep1
  ret void
}

; Offset = 256 * 16 = 4096 bytes = 0x1000
; The ADDXri encodes as #1, LSL #12 which equals 4096.
define void @stp_far_offset_q_4k_aligned(ptr %p, <2 x i64> %v0, <2 x i64> %v1) {
; ENABLED-LABEL: stp_far_offset_q_4k_aligned:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #1, lsl #12
; ENABLED-NEXT:    stp q0, q1, [x[[TMP]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_q_4k_aligned:
; DISABLED:       // %bb.0:
; DISABLED-NEXT:    str q0, [x0, #4096]
; DISABLED-NEXT:    str q1, [x0, #4112]
; DISABLED-NEXT:    ret
  %gep0 = getelementptr <2 x i64>, ptr %p, i64 256
  %gep1 = getelementptr <2 x i64>, ptr %p, i64 257
  store <2 x i64> %v0, ptr %gep0
  store <2 x i64> %v1, ptr %gep1
  ret void
}

; The preceding SUBXri that defines the pair's base register can be folded
; with the base-adjust ADDXri: sub x8, x0, #4096 + add x9, x8, #3696 merges
; into sub x8, x0, #400, saving one instruction.
define void @stp_far_offset_merge_sub(ptr %p, i32 %v0, i32 %v1) {
; ENABLED-LABEL: stp_far_offset_merge_sub:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[TMP:[0-9]+]], x0, #400
; ENABLED-NEXT:    stp w1, w2, [x[[TMP]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_merge_sub:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     stp w
; DISABLED-NOT:     stp x
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g1 = getelementptr i8, ptr %base, i64 3700
  store i32 %v0, ptr %g0
  store i32 %v1, ptr %g1
  ret void
}

; A third store reads the base register BETWEEN the two paired stores.
; The fold must be blocked: rewriting the SUBXri would change the base
; for the interleaved store, causing a miscompile.
define void @stp_far_offset_merge_blocked_interleaved(ptr %p, i32 %v0, i32 %vmid, i32 %v1) {
; ENABLED-LABEL: stp_far_offset_merge_blocked_interleaved:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[BASE:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED-NEXT:    add x[[ADJ:[0-9]+]], x[[BASE]], #3696
; ENABLED-NEXT:    str w[[RMID:[0-9]+]], [x[[BASE]], #200]
; ENABLED-NEXT:    stp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[ADJ]]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: stp_far_offset_merge_blocked_interleaved:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     stp w
; DISABLED-NOT:     stp x
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g_mid = getelementptr i8, ptr %base, i64 200
  %g1 = getelementptr i8, ptr %base, i64 3700
  store i32 %v0, ptr %g0
  store i32 %vmid, ptr %g_mid
  store i32 %v1, ptr %g1
  ret void
}

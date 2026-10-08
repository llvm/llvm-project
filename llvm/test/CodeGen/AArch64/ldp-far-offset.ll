; RUN: llc -mtriple=aarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefixes=CHECK,ENABLED
; RUN: llc -mtriple=aarch64 -verify-machineinstrs -aarch64-ldp-stp-base-adjust=0 < %s | FileCheck %s --check-prefixes=CHECK,DISABLED

define void @ldp_far_offset_i32(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_i32:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #400
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[TMP]]]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_i32:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp w
; DISABLED-NOT:     ldp x
; DISABLED:        ret
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

define void @ldp_far_offset_i64(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_i64:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #800
; ENABLED-NEXT:    ldp x[[R0:[0-9]+]], x[[R1:[0-9]+]], [x[[TMP]]]
; ENABLED-NEXT:    str x[[R0]]
; ENABLED-NEXT:    str x[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_i64:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp w
; DISABLED-NOT:     ldp x
; DISABLED:        ret
  %gep0 = getelementptr i64, ptr %p, i64 100
  %gep1 = getelementptr i64, ptr %p, i64 101
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

; The program-order-first load is at the higher offset (101), so the reused
; destination register lands in the second LDP destination slot (Rt2) and is
; also the adjusted base: `add Rt2, base, #imm; ldp Rt0, Rt2, [Rt2]`.  The
; backreference `x[[TMP]]` for both the second destination and the base
; makes this CHECK fail if a separate scratch is reintroduced.
define void @ldp_far_offset_i64_swapped(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_i64_swapped:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #800
; ENABLED-NEXT:    ldp x[[R0:[0-9]+]], x[[TMP]], [x[[TMP]]]
; ENABLED-NEXT:    str x[[TMP]]
; ENABLED-NEXT:    str x[[R0]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_i64_swapped:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp w
; DISABLED-NOT:     ldp x
; DISABLED:        ret
  %gep_hi = getelementptr i64, ptr %p, i64 101
  %gep_lo = getelementptr i64, ptr %p, i64 100
  %v_hi = load i64, ptr %gep_hi
  %v_lo = load i64, ptr %gep_lo
  store volatile i64 %v_hi, ptr poison
  store volatile i64 %v_lo, ptr poison
  ret void
}

; Regression guard for the GPR64 dest-reuse path: when the load destination
; is itself IP0/IP1 (X16/X17), reusing it as the adjusted base reintroduces
; the linker-veneer hazard the scavenger path avoids.  The dest-reuse path
; must skip X16/X17 and fall through to the X9..X15 scavenger.  The scratch
; capture `(1[0-5]|[0-9])` cannot match 16 or 17.
define void @ldp_far_offset_i64_x16_dst(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_i64_x16_dst:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[SCRATCH:(1[0-5]|[0-9])]], x0, #800
; ENABLED-NEXT:    ldp x16, x9, [x[[SCRATCH]]]
; ENABLED:         ret
;
; DISABLED-LABEL: ldp_far_offset_i64_x16_dst:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp w
; DISABLED-NOT:     ldp x
; DISABLED:        ret
  %gep0 = getelementptr i64, ptr %p, i64 100
  %gep1 = getelementptr i64, ptr %p, i64 101
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  %r0 = call i64 asm "", "={x16},0"(i64 %v0)
  %r1 = call i64 asm "", "={x9},0"(i64 %v1)
  store volatile i64 %r0, ptr poison
  store volatile i64 %r1, ptr poison
  ret void
}

define void @ldp_near_offset(ptr %p) {
; Near offsets within LDP range always pair regardless of the option.
; CHECK-LABEL: ldp_near_offset:
; CHECK:       // %bb.0:
; CHECK-NOT:     add
; CHECK:        ldp w{{[0-9]+}}, w{{[0-9]+}}, [x0, #40]
; CHECK:        ret
  %gep0 = getelementptr i32, ptr %p, i64 10
  %gep1 = getelementptr i32, ptr %p, i64 11
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

define void @ldp_far_offset_interleaved(ptr %p, ptr %q) {
; Intervening load from a different base between two far-offset loads.
; ENABLED-LABEL: ldp_far_offset_interleaved:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #400
; ENABLED:       ldp w{{[0-9]+}}, w{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: ldp_far_offset_interleaved:
; DISABLED-NOT:   ldp w
; DISABLED-NOT:   ldp x
; DISABLED:       ret
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  %v0 = load i32, ptr %gep0
  %v2 = load volatile i32, ptr %q
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; Offset = 1024 * 4 = 4096 bytes = 0x1000
; The ADDXri encodes as #1, LSL #12 which equals 4096.
define void @ldp_far_offset_4k_aligned_i32(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_4k_aligned_i32:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #1, lsl #12
; ENABLED:       ldp w{{[0-9]+}}, w{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: ldp_far_offset_4k_aligned_i32:
; DISABLED-NOT:   ldp w
; DISABLED-NOT:   ldp x
; DISABLED:       ret
  %gep0 = getelementptr i32, ptr %p, i64 1024
  %gep1 = getelementptr i32, ptr %p, i64 1025
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; Offset = 500 * 8 = 4000 bytes (fits in shift=0, <= 4095)
define void @ldp_far_offset_large_i64(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_large_i64:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #4000
; ENABLED:       ldp x{{[0-9]+}}, x{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: ldp_far_offset_large_i64:
; DISABLED-NOT:   ldp w
; DISABLED-NOT:   ldp x
; DISABLED:       ret
  %gep0 = getelementptr i64, ptr %p, i64 500
  %gep1 = getelementptr i64, ptr %p, i64 501
  %v0 = load i64, ptr %gep0
  %v1 = load i64, ptr %gep1
  store volatile i64 %v0, ptr poison
  store volatile i64 %v1, ptr poison
  ret void
}

define void @ldp_far_offset_q(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_q:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #1600
; ENABLED-NEXT:    ldp q0, q1, [x[[TMP]]]
; ENABLED-NEXT:    str q0, [x8]
; ENABLED-NEXT:    str q1, [x8]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_q:
; DISABLED:       // %bb.0:
; DISABLED-NEXT:    ldr q0, [x0, #1600]
; DISABLED-NEXT:    ldr q1, [x0, #1616]
; DISABLED-NEXT:    str q0, [x8]
; DISABLED-NEXT:    str q1, [x8]
; DISABLED-NEXT:    ret
  %gep0 = getelementptr <2 x i64>, ptr %p, i64 100
  %gep1 = getelementptr <2 x i64>, ptr %p, i64 101
  %v0 = load <2 x i64>, ptr %gep0
  %v1 = load <2 x i64>, ptr %gep1
  store volatile <2 x i64> %v0, ptr poison
  store volatile <2 x i64> %v1, ptr poison
  ret void
}

; Offset = 256 * 16 = 4096 bytes = 0x1000
; The ADDXri encodes as #1, LSL #12 which equals 4096.
define void @ldp_far_offset_q_4k_aligned(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_q_4k_aligned:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #1, lsl #12
; ENABLED-NEXT:    ldp q0, q1, [x[[TMP]]]
; ENABLED-NEXT:    str q0, [x8]
; ENABLED-NEXT:    str q1, [x8]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_q_4k_aligned:
; DISABLED:       // %bb.0:
; DISABLED-NEXT:    ldr q0, [x0, #4096]
; DISABLED-NEXT:    ldr q1, [x0, #4112]
; DISABLED-NEXT:    str q0, [x8]
; DISABLED-NEXT:    str q1, [x8]
; DISABLED-NEXT:    ret
  %gep0 = getelementptr <2 x i64>, ptr %p, i64 256
  %gep1 = getelementptr <2 x i64>, ptr %p, i64 257
  %v0 = load <2 x i64>, ptr %gep0
  %v1 = load <2 x i64>, ptr %gep1
  store volatile <2 x i64> %v0, ptr poison
  store volatile <2 x i64> %v1, ptr poison
  ret void
}

; Loop-invariant base inside a loop should still allow base-adjust.
define void @ldp_far_offset_loop_invariant(ptr %p, i32 %n) {
; ENABLED-LABEL: ldp_far_offset_loop_invariant:
; ENABLED:       add x[[TMP:[0-9]+]], x0, #400
; ENABLED:       ldp w{{[0-9]+}}, w{{[0-9]+}}, [x[[TMP]]]
;
; DISABLED-LABEL: ldp_far_offset_loop_invariant:
; DISABLED-NOT:   ldp w
; DISABLED-NOT:   ldp x
; DISABLED:       ret
entry:
  %gep0 = getelementptr i32, ptr %p, i64 100
  %gep1 = getelementptr i32, ptr %p, i64 101
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  %next = add i32 %i, 1
  %cond = icmp eq i32 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; Loop-variant base should not allow base-adjust: the base pointer changes
; each iteration (p+j), so the per-pair ADDXri would stay in the loop.
; Expect 2x ldr even with -aarch64-ldp-stp-base-adjust=1.
define void @ldp_far_offset_loop_variant(ptr %p, i32 %n) {
; CHECK-LABEL: ldp_far_offset_loop_variant:
; CHECK:       // %bb.0:
; CHECK-NOT:     add x{{[0-9]+}}, x{{[0-9]+}}, #400
; CHECK-NOT:     ldp w
; CHECK-NOT:     ldp x
; CHECK:         ret
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %idx = and i32 %i, 255
  %gep_base = getelementptr i32, ptr %p, i32 %idx
  %gep0 = getelementptr i32, ptr %gep_base, i64 100
  %gep1 = getelementptr i32, ptr %gep_base, i64 101
  %v0 = load i32, ptr %gep0
  %v1 = load i32, ptr %gep1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  %next = add i32 %i, 1
  %cond = icmp eq i32 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

; The preceding SUBXri that defines the pair's base register can be folded
; with the base-adjust ADDXri: sub x8, x0, #4096 + add x9, x8, #3696 merges
; into sub x8, x0, #400, saving one instruction.
define void @ldp_far_offset_merge_sub(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_sub:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[TMP:[0-9]+]], x0, #400
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[TMP]]]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_sub:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g1 = getelementptr i8, ptr %base, i64 3700
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; The preceding ADDXri can also be folded: add x8, x0, #96 + add x9, x8, #4000
; merges into add x8, x0, #4096 (lsl #12).
define void @ldp_far_offset_merge_add(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_add:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    add x[[TMP:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[TMP]]]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_add:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 96
  %g0 = getelementptr i8, ptr %base, i64 4000
  %g1 = getelementptr i8, ptr %base, i64 4004
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; When the preceding SUBXri and the base-adjust cancel out exactly, the def
; instruction is removed entirely and the pair uses the original source.
define void @ldp_far_offset_merge_zero(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_zero:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x0]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_zero:
; DISABLED:       // %bb.0:
; DISABLED:        ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 4096
  %g1 = getelementptr i8, ptr %base, i64 4100
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; The base register is also read by a third load (outside the pair), so the
; fold is blocked: rewriting the SUBXri would change the base for that load.
define void @ldp_far_offset_merge_blocked_extra_use(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_blocked_extra_use:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[BASE:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED-NEXT:    add x[[ADJ:[0-9]+]], x[[BASE]], #3696
; ENABLED-NEXT:    ldr w[[R2:[0-9]+]], [x[[BASE]], #3800]
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[ADJ]]]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    str w[[R2]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_blocked_extra_use:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g1 = getelementptr i8, ptr %base, i64 3700
  %g2 = getelementptr i8, ptr %base, i64 3800
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  %v2 = load i32, ptr %g2
  store volatile i32 %v0, ptr poison
  store volatile i32 %v1, ptr poison
  store volatile i32 %v2, ptr poison
  ret void
}

; A third load reads the base register BETWEEN the two paired loads (not
; before or after them). The fold must be blocked: rewriting the SUBXri
; would change the base for the interleaved load, causing a miscompile.
define void @ldp_far_offset_merge_blocked_interleaved(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_blocked_interleaved:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[BASE:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED-NEXT:    add x[[ADJ:[0-9]+]], x[[BASE]], #3696
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[ADJ]]]
; ENABLED-NEXT:    ldr w[[R2:[0-9]+]], [x[[BASE]], #200]
; ENABLED-NEXT:    str w[[R0]]
; ENABLED-NEXT:    str w[[R2]]
; ENABLED-NEXT:    str w[[R1]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_blocked_interleaved:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g_mid = getelementptr i8, ptr %base, i64 200
  %g1 = getelementptr i8, ptr %base, i64 3700
  %v0 = load i32, ptr %g0
  %vmid = load i32, ptr %g_mid
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr poison
  store volatile i32 %vmid, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

; After the pair, an instruction reads the low 32 bits of the base register
; (w8 = sub-register of x8). The pair's destinations (w9, w10) do NOT alias
; x8, so w8 still holds the changed base value after the pair — not a loaded
; value. The fold must be blocked, otherwise the post-pair read gets the
; changed base instead of the original.
define i32 @ldp_far_offset_merge_blocked_subreg_after(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_blocked_subreg_after:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[BASE:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED-NEXT:    add x[[ADJ:[0-9]+]], x[[BASE]], #3696
; ENABLED-NEXT:    ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[ADJ]]]
; ENABLED-NEXT:    add w[[R0]], w[[R0]], w[[R1]]
; ENABLED-NEXT:    add w0, w[[R0]], w[[BASE]]
; ENABLED-NEXT:    ret
;
; DISABLED-LABEL: ldp_far_offset_merge_blocked_subreg_after:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g1 = getelementptr i8, ptr %base, i64 3700
  %v0 = load i32, ptr %g0
  %v1 = load i32, ptr %g1
  %baseint = ptrtoint ptr %base to i64
  %trunc = trunc i64 %baseint to i32
  %s = add i32 %v0, %v1
  %s2 = add i32 %s, %trunc
  ret i32 %s2
}

; The base-adjust def is not immediately before the first paired instruction
; (an extra instruction sits between them), so the fold must be skipped: it
; falls back to inserting a separate ADDXri. This is a known, accepted
; trade-off of the conservative adjacency-only design. The volatile load of
; the same base between the two paired loads pushes a LDR between the SUBXri
; and the first paired LDR, breaking adjacency.
define void @ldp_far_offset_merge_blocked_non_adjacent(ptr %p) {
; ENABLED-LABEL: ldp_far_offset_merge_blocked_non_adjacent:
; ENABLED:       // %bb.0:
; ENABLED-NEXT:    sub x[[BASE:[0-9]+]], x0, #1, lsl #12 // =4096
; ENABLED:         add x[[ADJ:[0-9]+]], x[[BASE]], #3696
; ENABLED:         ldp w[[R0:[0-9]+]], w[[R1:[0-9]+]], [x[[ADJ]]]
; ENABLED:         ldr w[[R2:[0-9]+]], [x[[BASE]]]
; ENABLED:         str w[[R0]]
; ENABLED:         str w[[R2]]
; ENABLED:         str w[[R1]]
; ENABLED:         ret
;
; DISABLED-LABEL: ldp_far_offset_merge_blocked_non_adjacent:
; DISABLED:       // %bb.0:
; DISABLED-NOT:     ldp
; DISABLED:        ret
  %base = getelementptr i8, ptr %p, i64 -4096
  %g0 = getelementptr i8, ptr %base, i64 3696
  %g1 = getelementptr i8, ptr %base, i64 3700
  %v0 = load i32, ptr %g0
  %v0b = load volatile i32, ptr %base
  %v1 = load i32, ptr %g1
  store volatile i32 %v0, ptr poison
  store volatile i32 %v0b, ptr poison
  store volatile i32 %v1, ptr poison
  ret void
}

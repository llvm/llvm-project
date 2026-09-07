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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
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
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
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
  store volatile i64 %v_hi, ptr undef
  store volatile i64 %v_lo, ptr undef
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
  store volatile i64 %r0, ptr undef
  store volatile i64 %r1, ptr undef
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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
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
  store volatile i64 %v0, ptr undef
  store volatile i64 %v1, ptr undef
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
  store volatile <2 x i64> %v0, ptr undef
  store volatile <2 x i64> %v1, ptr undef
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
  store volatile <2 x i64> %v0, ptr undef
  store volatile <2 x i64> %v1, ptr undef
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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
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
  store volatile i32 %v0, ptr undef
  store volatile i32 %v1, ptr undef
  %next = add i32 %i, 1
  %cond = icmp eq i32 %next, %n
  br i1 %cond, label %exit, label %loop

exit:
  ret void
}

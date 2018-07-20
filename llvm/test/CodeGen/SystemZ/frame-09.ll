; Test the handling of the frame pointer (%r11).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -disable-fp-elim | FileCheck %s

; We should always initialise %r11 when FP elimination is disabled.
; We don't need to allocate any more than the caller-provided 160-byte
; area though.
define i32 @f1(i32 %x) {
; CHECK-LABEL: f1:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r15, -40
; CHECK-NOT: ag
; CHECK: lgr %r11, %r15
; CHECK: .cfi_def_cfa_register %r11
; CHECK: lmg %r11, %r15, 88(%r11)
; CHECK: br %r14
  %y = add i32 %x, 1
  ret i32 %y
}

; Make sure that frame accesses after the initial allocation are relative
; to %r11 rather than %r15.
define void @f2(i64 %x) {
; CHECK-LABEL: f2:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r15, -40
; CHECK: aghi %r15, -168
; CHECK: .cfi_def_cfa_offset 328
; CHECK: lgr %r11, %r15
; CHECK: .cfi_def_cfa_register %r11
; CHECK: stg %r2, 160(%r11)
; CHECK: lmg %r11, %r15, 256(%r11)
; CHECK: br %r14
  %y = alloca i64, align 8
  store volatile i64 %x, i64* %y
  ret void
}

; This function should require all GPRs but no other spill slots.
; It shouldn't need to allocate its own frame.
define void @f3(i32 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK-NOT: %r15
; CHECK-NOT: %r11
; CHECK: .cfi_offset %r6, -112
; CHECK: .cfi_offset %r7, -104
; CHECK: .cfi_offset %r8, -96
; CHECK: .cfi_offset %r9, -88
; CHECK: .cfi_offset %r10, -80
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r12, -64
; CHECK: .cfi_offset %r13, -56
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; CHECK-NOT: ag
; CHECK: lgr %r11, %r15
; CHECK: .cfi_def_cfa_register %r11
; ...main function body...
; CHECK-NOT: %r15
; CHECK-NOT: %r11
; CHECK: st {{.*}}, 4(%r2)
; CHECK: lmg %r6, %r15, 48(%r11)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l3 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l6 = load volatile i32, i32 *%ptr
  %l7 = load volatile i32, i32 *%ptr
  %l8 = load volatile i32, i32 *%ptr
  %l9 = load volatile i32, i32 *%ptr
  %l10 = load volatile i32, i32 *%ptr
  %l12 = load volatile i32, i32 *%ptr
  %l13 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add3 = add i32 %l3, %add1
  %add4 = add i32 %l4, %add3
  %add5 = add i32 %l5, %add4
  %add6 = add i32 %l6, %add5
  %add7 = add i32 %l7, %add6
  %add8 = add i32 %l8, %add7
  %add9 = add i32 %l9, %add8
  %add10 = add i32 %l10, %add9
  %add12 = add i32 %l12, %add10
  %add13 = add i32 %l13, %add12
  %add14 = add i32 %l14, %add13
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add3, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add6, i32 *%ptr
  store volatile i32 %add7, i32 *%ptr
  store volatile i32 %add8, i32 *%ptr
  store volatile i32 %add9, i32 *%ptr
  store volatile i32 %add10, i32 *%ptr
  store volatile i32 %add12, i32 *%ptr
  store volatile i32 %add13, i32 *%ptr
  %final = getelementptr i32, i32 *%ptr, i32 1
  store volatile i32 %add14, i32 *%final
  ret void
}

; The largest frame for which the LMG is in range.  This frame has two
; emergency spill slots at 160(%r11), so create a frame of size 524192
; by allocating (524192 - 176) / 8 = 65502 doublewords.
define void @f4(i64 %x) {
; CHECK-LABEL: f4:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524192
; CHECK: .cfi_def_cfa_offset 524352
; CHECK: lgr %r11, %r15
; CHECK: .cfi_def_cfa_register %r11
; CHECK: stg %r2, 176(%r11)
; CHECK-NOT: ag
; CHECK: lmg %r11, %r15, 524280(%r11)
; CHECK: br %r14
  %y = alloca [65502 x i64], align 8
  %ptr = getelementptr inbounds [65502 x i64], [65502 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; The next frame size larger than f4.
define void @f5(i64 %x) {
; CHECK-LABEL: f5:
; CHECK: stmg %r11, %r15, 88(%r15)
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r15, -40
; CHECK: agfi %r15, -524200
; CHECK: .cfi_def_cfa_offset 524360
; CHECK: lgr %r11, %r15
; CHECK: .cfi_def_cfa_register %r11
; CHECK: stg %r2, 176(%r11)
; CHECK: aghi %r11, 8
; CHECK: lmg %r11, %r15, 524280(%r11)
; CHECK: br %r14
  %y = alloca [65503 x i64], align 8
  %ptr = getelementptr inbounds [65503 x i64], [65503 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %ptr
  ret void
}

; The tests above establish that %r11 is handled like %r15 for LMG.
; Rely on the %r15-based tests in frame-08.ll for other cases.

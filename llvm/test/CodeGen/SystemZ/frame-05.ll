; Test saving and restoring of call-saved GPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This function should require all GPRs, but no other spill slots.  The caller
; allocates room for the GPR save slots, so we shouldn't need to allocate any
; extra space.
;
; The function only modifies the low 32 bits of each register, which in
; itself would allow STM and LM to be used instead of STMG and LMG.
; However, the ABI defines the offset of each register, so we always
; use the 64-bit form.
;
; Use a different address for the final store, so that we can check that
; %r15 isn't referenced again until after that.
define void @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: stmg %r6, %r15, 48(%r15)
; CHECK-NOT: %r15
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
; ...main function body...
; CHECK-NOT: %r15
; CHECK: st {{.*}}, 4(%r2)
; CHECK: lmg %r6, %r15, 48(%r15)
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
  %l11 = load volatile i32, i32 *%ptr
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
  %add11 = add i32 %l11, %add10
  %add12 = add i32 %l12, %add11
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
  store volatile i32 %add11, i32 *%ptr
  store volatile i32 %add12, i32 *%ptr
  store volatile i32 %add13, i32 *%ptr
  %final = getelementptr i32, i32 *%ptr, i32 1
  store volatile i32 %add14, i32 *%final
  ret void
}

; Like f1, but requires one fewer GPR.  We allocate the call-saved GPRs
; from %r14 down, so that the STMG/LMG sequences aren't any longer than
; they need to be.
define void @f2(i32 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: stmg %r7, %r15, 56(%r15)
; CHECK-NOT: %r15
; CHECK: .cfi_offset %r7, -104
; CHECK: .cfi_offset %r8, -96
; CHECK: .cfi_offset %r9, -88
; CHECK: .cfi_offset %r10, -80
; CHECK: .cfi_offset %r11, -72
; CHECK: .cfi_offset %r12, -64
; CHECK: .cfi_offset %r13, -56
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; ...main function body...
; CHECK-NOT: %r15
; CHECK-NOT: %r6
; CHECK: st {{.*}}, 4(%r2)
; CHECK: lmg %r7, %r15, 56(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l3 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l7 = load volatile i32, i32 *%ptr
  %l8 = load volatile i32, i32 *%ptr
  %l9 = load volatile i32, i32 *%ptr
  %l10 = load volatile i32, i32 *%ptr
  %l11 = load volatile i32, i32 *%ptr
  %l12 = load volatile i32, i32 *%ptr
  %l13 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add3 = add i32 %l3, %add1
  %add4 = add i32 %l4, %add3
  %add5 = add i32 %l5, %add4
  %add7 = add i32 %l7, %add5
  %add8 = add i32 %l8, %add7
  %add9 = add i32 %l9, %add8
  %add10 = add i32 %l10, %add9
  %add11 = add i32 %l11, %add10
  %add12 = add i32 %l12, %add11
  %add13 = add i32 %l13, %add12
  %add14 = add i32 %l14, %add13
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add3, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  store volatile i32 %add7, i32 *%ptr
  store volatile i32 %add8, i32 *%ptr
  store volatile i32 %add9, i32 *%ptr
  store volatile i32 %add10, i32 *%ptr
  store volatile i32 %add11, i32 *%ptr
  store volatile i32 %add12, i32 *%ptr
  store volatile i32 %add13, i32 *%ptr
  %final = getelementptr i32, i32 *%ptr, i32 1
  store volatile i32 %add14, i32 *%final
  ret void
}

; Like f1, but only needs one call-saved GPR, which ought to be %r14.
define void @f3(i32 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: stmg %r14, %r15, 112(%r15)
; CHECK-NOT: %r15
; CHECK: .cfi_offset %r14, -48
; CHECK: .cfi_offset %r15, -40
; ...main function body...
; CHECK-NOT: %r15
; CHECK-NOT: %r6
; CHECK-NOT: %r7
; CHECK-NOT: %r8
; CHECK-NOT: %r9
; CHECK-NOT: %r10
; CHECK-NOT: %r11
; CHECK-NOT: %r12
; CHECK-NOT: %r13
; CHECK: st {{.*}}, 4(%r2)
; CHECK: lmg %r14, %r15, 112(%r15)
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l3 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %l14 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add3 = add i32 %l3, %add1
  %add4 = add i32 %l4, %add3
  %add5 = add i32 %l5, %add4
  %add14 = add i32 %l14, %add5
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add3, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  store volatile i32 %add5, i32 *%ptr
  %final = getelementptr i32, i32 *%ptr, i32 1
  store volatile i32 %add14, i32 *%final
  ret void
}

; This function should use all call-clobbered GPRs but no call-saved ones.
; It shouldn't need to touch the stack at all.
define void @f4(i32 *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r15
; CHECK-NOT: %r6
; CHECK-NOT: %r7
; CHECK-NOT: %r8
; CHECK-NOT: %r9
; CHECK-NOT: %r10
; CHECK-NOT: %r11
; CHECK-NOT: %r12
; CHECK-NOT: %r13
; CHECK: br %r14
  %l0 = load volatile i32, i32 *%ptr
  %l1 = load volatile i32, i32 *%ptr
  %l3 = load volatile i32, i32 *%ptr
  %l4 = load volatile i32, i32 *%ptr
  %l5 = load volatile i32, i32 *%ptr
  %add0 = add i32 %l0, %l0
  %add1 = add i32 %l1, %add0
  %add3 = add i32 %l3, %add1
  %add4 = add i32 %l4, %add3
  %add5 = add i32 %l5, %add4
  store volatile i32 %add0, i32 *%ptr
  store volatile i32 %add1, i32 *%ptr
  store volatile i32 %add3, i32 *%ptr
  store volatile i32 %add4, i32 *%ptr
  %final = getelementptr i32, i32 *%ptr, i32 1
  store volatile i32 %add5, i32 *%final
  ret void
}

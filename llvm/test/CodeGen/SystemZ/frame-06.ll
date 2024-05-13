; Like frame-05.ll, but with i64s rather than i32s.  Internally this
; uses a different register class, but the set of saved and restored
; registers should be the same.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This function should require all GPRs, but no other spill slots.  The caller
; allocates room for the GPR save slots, so we shouldn't need to allocate any
; extra space.
;
; Use a different address for the final store, so that we can check that
; %r15 isn't referenced again until after that.
define void @f1(ptr %ptr) {
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
; CHECK: stg {{.*}}, 8(%r2)
; CHECK: lmg %r6, %r15, 48(%r15)
; CHECK: br %r14
  %l0 = load volatile i64, ptr %ptr
  %l1 = load volatile i64, ptr %ptr
  %l3 = load volatile i64, ptr %ptr
  %l4 = load volatile i64, ptr %ptr
  %l5 = load volatile i64, ptr %ptr
  %l6 = load volatile i64, ptr %ptr
  %l7 = load volatile i64, ptr %ptr
  %l8 = load volatile i64, ptr %ptr
  %l9 = load volatile i64, ptr %ptr
  %l10 = load volatile i64, ptr %ptr
  %l11 = load volatile i64, ptr %ptr
  %l12 = load volatile i64, ptr %ptr
  %l13 = load volatile i64, ptr %ptr
  %l14 = load volatile i64, ptr %ptr
  %add0 = add i64 %l0, %l0
  %add1 = add i64 %l1, %add0
  %add3 = add i64 %l3, %add1
  %add4 = add i64 %l4, %add3
  %add5 = add i64 %l5, %add4
  %add6 = add i64 %l6, %add5
  %add7 = add i64 %l7, %add6
  %add8 = add i64 %l8, %add7
  %add9 = add i64 %l9, %add8
  %add10 = add i64 %l10, %add9
  %add11 = add i64 %l11, %add10
  %add12 = add i64 %l12, %add11
  %add13 = add i64 %l13, %add12
  %add14 = add i64 %l14, %add13
  store volatile i64 %add0, ptr %ptr
  store volatile i64 %add1, ptr %ptr
  store volatile i64 %add3, ptr %ptr
  store volatile i64 %add4, ptr %ptr
  store volatile i64 %add5, ptr %ptr
  store volatile i64 %add6, ptr %ptr
  store volatile i64 %add7, ptr %ptr
  store volatile i64 %add8, ptr %ptr
  store volatile i64 %add9, ptr %ptr
  store volatile i64 %add10, ptr %ptr
  store volatile i64 %add11, ptr %ptr
  store volatile i64 %add12, ptr %ptr
  store volatile i64 %add13, ptr %ptr
  %final = getelementptr i64, ptr %ptr, i64 1
  store volatile i64 %add14, ptr %final
  ret void
}

; Like f1, but requires one fewer GPR.  We allocate the call-saved GPRs
; from %r14 down, so that the STMG/LMG sequences aren't any longer than
; they need to be.
define void @f2(ptr %ptr) {
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
; CHECK: stg {{.*}}, 8(%r2)
; CHECK: lmg %r7, %r15, 56(%r15)
; CHECK: br %r14
  %l0 = load volatile i64, ptr %ptr
  %l1 = load volatile i64, ptr %ptr
  %l3 = load volatile i64, ptr %ptr
  %l4 = load volatile i64, ptr %ptr
  %l5 = load volatile i64, ptr %ptr
  %l7 = load volatile i64, ptr %ptr
  %l8 = load volatile i64, ptr %ptr
  %l9 = load volatile i64, ptr %ptr
  %l10 = load volatile i64, ptr %ptr
  %l11 = load volatile i64, ptr %ptr
  %l12 = load volatile i64, ptr %ptr
  %l13 = load volatile i64, ptr %ptr
  %l14 = load volatile i64, ptr %ptr
  %add0 = add i64 %l0, %l0
  %add1 = add i64 %l1, %add0
  %add3 = add i64 %l3, %add1
  %add4 = add i64 %l4, %add3
  %add5 = add i64 %l5, %add4
  %add7 = add i64 %l7, %add5
  %add8 = add i64 %l8, %add7
  %add9 = add i64 %l9, %add8
  %add10 = add i64 %l10, %add9
  %add11 = add i64 %l11, %add10
  %add12 = add i64 %l12, %add11
  %add13 = add i64 %l13, %add12
  %add14 = add i64 %l14, %add13
  store volatile i64 %add0, ptr %ptr
  store volatile i64 %add1, ptr %ptr
  store volatile i64 %add3, ptr %ptr
  store volatile i64 %add4, ptr %ptr
  store volatile i64 %add5, ptr %ptr
  store volatile i64 %add7, ptr %ptr
  store volatile i64 %add8, ptr %ptr
  store volatile i64 %add9, ptr %ptr
  store volatile i64 %add10, ptr %ptr
  store volatile i64 %add11, ptr %ptr
  store volatile i64 %add12, ptr %ptr
  store volatile i64 %add13, ptr %ptr
  %final = getelementptr i64, ptr %ptr, i64 1
  store volatile i64 %add14, ptr %final
  ret void
}

; Like f1, but only needs one call-saved GPR, which ought to be %r14.
define void @f3(ptr %ptr) {
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
; CHECK: stg {{.*}}, 8(%r2)
; CHECK: lmg %r14, %r15, 112(%r15)
; CHECK: br %r14
  %l0 = load volatile i64, ptr %ptr
  %l1 = load volatile i64, ptr %ptr
  %l3 = load volatile i64, ptr %ptr
  %l4 = load volatile i64, ptr %ptr
  %l5 = load volatile i64, ptr %ptr
  %l14 = load volatile i64, ptr %ptr
  %add0 = add i64 %l0, %l0
  %add1 = add i64 %l1, %add0
  %add3 = add i64 %l3, %add1
  %add4 = add i64 %l4, %add3
  %add5 = add i64 %l5, %add4
  %add14 = add i64 %l14, %add5
  store volatile i64 %add0, ptr %ptr
  store volatile i64 %add1, ptr %ptr
  store volatile i64 %add3, ptr %ptr
  store volatile i64 %add4, ptr %ptr
  store volatile i64 %add5, ptr %ptr
  %final = getelementptr i64, ptr %ptr, i64 1
  store volatile i64 %add14, ptr %final
  ret void
}

; This function should use all call-clobbered GPRs but no call-saved ones.
; It shouldn't need to touch the stack at all.
define void @f4(ptr %ptr) {
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
  %l0 = load volatile i64, ptr %ptr
  %l1 = load volatile i64, ptr %ptr
  %l3 = load volatile i64, ptr %ptr
  %l4 = load volatile i64, ptr %ptr
  %l5 = load volatile i64, ptr %ptr
  %add0 = add i64 %l0, %l0
  %add1 = add i64 %l1, %add0
  %add3 = add i64 %l3, %add1
  %add4 = add i64 %l4, %add3
  %add5 = add i64 %l5, %add4
  store volatile i64 %add0, ptr %ptr
  store volatile i64 %add1, ptr %ptr
  store volatile i64 %add3, ptr %ptr
  store volatile i64 %add4, ptr %ptr
  %final = getelementptr i64, ptr %ptr, i64 1
  store volatile i64 %add5, ptr %final
  ret void
}

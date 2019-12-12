; Like frame-02.ll, but with doubles rather than floats.  Internally this
; uses a different register class, but the set of saved and restored
; registers should be the same.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; This function should require all FPRs, but no other spill slots.
; We need to save and restore 8 of the 16 FPRs, so the frame size
; should be exactly 8 * 8 = 64.  The CFA offset is 160
; (the caller-allocated part of the frame) + 64.
define void @f1(double *%ptr) {
; CHECK-LABEL: f1:
; CHECK: aghi %r15, -64
; CHECK: .cfi_def_cfa_offset 224
; CHECK: std %f8, 56(%r15)
; CHECK: std %f9, 48(%r15)
; CHECK: std %f10, 40(%r15)
; CHECK: std %f11, 32(%r15)
; CHECK: std %f12, 24(%r15)
; CHECK: std %f13, 16(%r15)
; CHECK: std %f14, 8(%r15)
; CHECK: std %f15, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f9, -176
; CHECK: .cfi_offset %f10, -184
; CHECK: .cfi_offset %f11, -192
; CHECK: .cfi_offset %f12, -200
; CHECK: .cfi_offset %f13, -208
; CHECK: .cfi_offset %f14, -216
; CHECK: .cfi_offset %f15, -224
; ...main function body...
; CHECK: ld %f8, 56(%r15)
; CHECK: ld %f9, 48(%r15)
; CHECK: ld %f10, 40(%r15)
; CHECK: ld %f11, 32(%r15)
; CHECK: ld %f12, 24(%r15)
; CHECK: ld %f13, 16(%r15)
; CHECK: ld %f14, 8(%r15)
; CHECK: ld %f15, 0(%r15)
; CHECK: aghi %r15, 64
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %l8 = load volatile double, double *%ptr
  %l9 = load volatile double, double *%ptr
  %l10 = load volatile double, double *%ptr
  %l11 = load volatile double, double *%ptr
  %l12 = load volatile double, double *%ptr
  %l13 = load volatile double, double *%ptr
  %l14 = load volatile double, double *%ptr
  %l15 = load volatile double, double *%ptr
  %add0 = fadd double %l0, %l0
  %add1 = fadd double %l1, %add0
  %add2 = fadd double %l2, %add1
  %add3 = fadd double %l3, %add2
  %add4 = fadd double %l4, %add3
  %add5 = fadd double %l5, %add4
  %add6 = fadd double %l6, %add5
  %add7 = fadd double %l7, %add6
  %add8 = fadd double %l8, %add7
  %add9 = fadd double %l9, %add8
  %add10 = fadd double %l10, %add9
  %add11 = fadd double %l11, %add10
  %add12 = fadd double %l12, %add11
  %add13 = fadd double %l13, %add12
  %add14 = fadd double %l14, %add13
  %add15 = fadd double %l15, %add14
  store volatile double %add0, double *%ptr
  store volatile double %add1, double *%ptr
  store volatile double %add2, double *%ptr
  store volatile double %add3, double *%ptr
  store volatile double %add4, double *%ptr
  store volatile double %add5, double *%ptr
  store volatile double %add6, double *%ptr
  store volatile double %add7, double *%ptr
  store volatile double %add8, double *%ptr
  store volatile double %add9, double *%ptr
  store volatile double %add10, double *%ptr
  store volatile double %add11, double *%ptr
  store volatile double %add12, double *%ptr
  store volatile double %add13, double *%ptr
  store volatile double %add14, double *%ptr
  store volatile double %add15, double *%ptr
  ret void
}

; Like f1, but requires one fewer FPR.  We allocate in numerical order,
; so %f15 is the one that gets dropped.
define void @f2(double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -56
; CHECK: .cfi_def_cfa_offset 216
; CHECK: std %f8, 48(%r15)
; CHECK: std %f9, 40(%r15)
; CHECK: std %f10, 32(%r15)
; CHECK: std %f11, 24(%r15)
; CHECK: std %f12, 16(%r15)
; CHECK: std %f13, 8(%r15)
; CHECK: std %f14, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f9, -176
; CHECK: .cfi_offset %f10, -184
; CHECK: .cfi_offset %f11, -192
; CHECK: .cfi_offset %f12, -200
; CHECK: .cfi_offset %f13, -208
; CHECK: .cfi_offset %f14, -216
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 48(%r15)
; CHECK: ld %f9, 40(%r15)
; CHECK: ld %f10, 32(%r15)
; CHECK: ld %f11, 24(%r15)
; CHECK: ld %f12, 16(%r15)
; CHECK: ld %f13, 8(%r15)
; CHECK: ld %f14, 0(%r15)
; CHECK: aghi %r15, 56
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %l8 = load volatile double, double *%ptr
  %l9 = load volatile double, double *%ptr
  %l10 = load volatile double, double *%ptr
  %l11 = load volatile double, double *%ptr
  %l12 = load volatile double, double *%ptr
  %l13 = load volatile double, double *%ptr
  %l14 = load volatile double, double *%ptr
  %add0 = fadd double %l0, %l0
  %add1 = fadd double %l1, %add0
  %add2 = fadd double %l2, %add1
  %add3 = fadd double %l3, %add2
  %add4 = fadd double %l4, %add3
  %add5 = fadd double %l5, %add4
  %add6 = fadd double %l6, %add5
  %add7 = fadd double %l7, %add6
  %add8 = fadd double %l8, %add7
  %add9 = fadd double %l9, %add8
  %add10 = fadd double %l10, %add9
  %add11 = fadd double %l11, %add10
  %add12 = fadd double %l12, %add11
  %add13 = fadd double %l13, %add12
  %add14 = fadd double %l14, %add13
  store volatile double %add0, double *%ptr
  store volatile double %add1, double *%ptr
  store volatile double %add2, double *%ptr
  store volatile double %add3, double *%ptr
  store volatile double %add4, double *%ptr
  store volatile double %add5, double *%ptr
  store volatile double %add6, double *%ptr
  store volatile double %add7, double *%ptr
  store volatile double %add8, double *%ptr
  store volatile double %add9, double *%ptr
  store volatile double %add10, double *%ptr
  store volatile double %add11, double *%ptr
  store volatile double %add12, double *%ptr
  store volatile double %add13, double *%ptr
  store volatile double %add14, double *%ptr
  ret void
}

; Like f1, but should require only one call-saved FPR.
define void @f3(double *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -8
; CHECK: .cfi_def_cfa_offset 168
; CHECK: std %f8, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK-NOT: %f9
; CHECK-NOT: %f10
; CHECK-NOT: %f11
; CHECK-NOT: %f12
; CHECK-NOT: %f13
; CHECK-NOT: %f14
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 0(%r15)
; CHECK: aghi %r15, 8
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %l8 = load volatile double, double *%ptr
  %add0 = fadd double %l0, %l0
  %add1 = fadd double %l1, %add0
  %add2 = fadd double %l2, %add1
  %add3 = fadd double %l3, %add2
  %add4 = fadd double %l4, %add3
  %add5 = fadd double %l5, %add4
  %add6 = fadd double %l6, %add5
  %add7 = fadd double %l7, %add6
  %add8 = fadd double %l8, %add7
  store volatile double %add0, double *%ptr
  store volatile double %add1, double *%ptr
  store volatile double %add2, double *%ptr
  store volatile double %add3, double *%ptr
  store volatile double %add4, double *%ptr
  store volatile double %add5, double *%ptr
  store volatile double %add6, double *%ptr
  store volatile double %add7, double *%ptr
  store volatile double %add8, double *%ptr
  ret void
}

; This function should use all call-clobbered FPRs but no call-saved ones.
; It shouldn't need to create a frame.
define void @f4(double *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r15
; CHECK-NOT: %f8
; CHECK-NOT: %f9
; CHECK-NOT: %f10
; CHECK-NOT: %f11
; CHECK-NOT: %f12
; CHECK-NOT: %f13
; CHECK-NOT: %f14
; CHECK-NOT: %f15
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %add0 = fadd double %l0, %l0
  %add1 = fadd double %l1, %add0
  %add2 = fadd double %l2, %add1
  %add3 = fadd double %l3, %add2
  %add4 = fadd double %l4, %add3
  %add5 = fadd double %l5, %add4
  %add6 = fadd double %l6, %add5
  %add7 = fadd double %l7, %add6
  store volatile double %add0, double *%ptr
  store volatile double %add1, double *%ptr
  store volatile double %add2, double *%ptr
  store volatile double %add3, double *%ptr
  store volatile double %add4, double *%ptr
  store volatile double %add5, double *%ptr
  store volatile double %add6, double *%ptr
  store volatile double %add7, double *%ptr
  ret void
}

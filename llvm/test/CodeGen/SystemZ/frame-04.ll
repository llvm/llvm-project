; Like frame-02.ll, but with long doubles rather than floats.  Some of the
; cases are slightly different because we need to allocate pairs of FPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This function should require all FPRs, but no other spill slots.
; We need to save and restore 8 of the 16 FPRs, so the frame size
; should be exactly 8 * 8 = 64.  The CFA offset is 160
; (the caller-allocated part of the frame) + 64.
define void @f1(fp128 *%ptr) {
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
  %l0 = load volatile fp128, fp128 *%ptr
  %l1 = load volatile fp128, fp128 *%ptr
  %l4 = load volatile fp128, fp128 *%ptr
  %l5 = load volatile fp128, fp128 *%ptr
  %l8 = load volatile fp128, fp128 *%ptr
  %l9 = load volatile fp128, fp128 *%ptr
  %l12 = load volatile fp128, fp128 *%ptr
  %l13 = load volatile fp128, fp128 *%ptr
  %add0 = fadd fp128 %l0, %l0
  %add1 = fadd fp128 %l1, %add0
  %add4 = fadd fp128 %l4, %add1
  %add5 = fadd fp128 %l5, %add4
  %add8 = fadd fp128 %l8, %add5
  %add9 = fadd fp128 %l9, %add8
  %add12 = fadd fp128 %l12, %add9
  %add13 = fadd fp128 %l13, %add12
  store volatile fp128 %add0, fp128 *%ptr
  store volatile fp128 %add1, fp128 *%ptr
  store volatile fp128 %add4, fp128 *%ptr
  store volatile fp128 %add5, fp128 *%ptr
  store volatile fp128 %add8, fp128 *%ptr
  store volatile fp128 %add9, fp128 *%ptr
  store volatile fp128 %add12, fp128 *%ptr
  store volatile fp128 %add13, fp128 *%ptr
  ret void
}

; Like f1, but requires one fewer FPR pair.  We allocate in numerical order,
; so %f13+%f15 is the pair that gets dropped.
define void @f2(fp128 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -48
; CHECK: .cfi_def_cfa_offset 208
; CHECK: std %f8, 40(%r15)
; CHECK: std %f9, 32(%r15)
; CHECK: std %f10, 24(%r15)
; CHECK: std %f11, 16(%r15)
; CHECK: std %f12, 8(%r15)
; CHECK: std %f14, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f9, -176
; CHECK: .cfi_offset %f10, -184
; CHECK: .cfi_offset %f11, -192
; CHECK: .cfi_offset %f12, -200
; CHECK: .cfi_offset %f14, -208
; CHECK-NOT: %f13
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 40(%r15)
; CHECK: ld %f9, 32(%r15)
; CHECK: ld %f10, 24(%r15)
; CHECK: ld %f11, 16(%r15)
; CHECK: ld %f12, 8(%r15)
; CHECK: ld %f14, 0(%r15)
; CHECK: aghi %r15, 48
; CHECK: br %r14
  %l0 = load volatile fp128, fp128 *%ptr
  %l1 = load volatile fp128, fp128 *%ptr
  %l4 = load volatile fp128, fp128 *%ptr
  %l5 = load volatile fp128, fp128 *%ptr
  %l8 = load volatile fp128, fp128 *%ptr
  %l9 = load volatile fp128, fp128 *%ptr
  %l12 = load volatile fp128, fp128 *%ptr
  %add0 = fadd fp128 %l0, %l0
  %add1 = fadd fp128 %l1, %add0
  %add4 = fadd fp128 %l4, %add1
  %add5 = fadd fp128 %l5, %add4
  %add8 = fadd fp128 %l8, %add5
  %add9 = fadd fp128 %l9, %add8
  %add12 = fadd fp128 %l12, %add9
  store volatile fp128 %add0, fp128 *%ptr
  store volatile fp128 %add1, fp128 *%ptr
  store volatile fp128 %add4, fp128 *%ptr
  store volatile fp128 %add5, fp128 *%ptr
  store volatile fp128 %add8, fp128 *%ptr
  store volatile fp128 %add9, fp128 *%ptr
  store volatile fp128 %add12, fp128 *%ptr
  ret void
}

; Like f1, but requires only one call-saved FPR pair.  We allocate in
; numerical order so the pair should be %f8+%f10.
define void @f3(fp128 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -16
; CHECK: .cfi_def_cfa_offset 176
; CHECK: std %f8, 8(%r15)
; CHECK: std %f10, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f10, -176
; CHECK-NOT: %f9
; CHECK-NOT: %f11
; CHECK-NOT: %f12
; CHECK-NOT: %f13
; CHECK-NOT: %f14
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 8(%r15)
; CHECK: ld %f10, 0(%r15)
; CHECK: aghi %r15, 16
; CHECK: br %r14
  %l0 = load volatile fp128, fp128 *%ptr
  %l1 = load volatile fp128, fp128 *%ptr
  %l4 = load volatile fp128, fp128 *%ptr
  %l5 = load volatile fp128, fp128 *%ptr
  %l8 = load volatile fp128, fp128 *%ptr
  %add0 = fadd fp128 %l0, %l0
  %add1 = fadd fp128 %l1, %add0
  %add4 = fadd fp128 %l4, %add1
  %add5 = fadd fp128 %l5, %add4
  %add8 = fadd fp128 %l8, %add5
  store volatile fp128 %add0, fp128 *%ptr
  store volatile fp128 %add1, fp128 *%ptr
  store volatile fp128 %add4, fp128 *%ptr
  store volatile fp128 %add5, fp128 *%ptr
  store volatile fp128 %add8, fp128 *%ptr
  ret void
}

; This function should use all call-clobbered FPRs but no call-saved ones.
; It shouldn't need to create a frame.
define void @f4(fp128 *%ptr) {
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
  %l0 = load volatile fp128, fp128 *%ptr
  %l1 = load volatile fp128, fp128 *%ptr
  %l4 = load volatile fp128, fp128 *%ptr
  %l5 = load volatile fp128, fp128 *%ptr
  %add0 = fadd fp128 %l0, %l0
  %add1 = fadd fp128 %l1, %add0
  %add4 = fadd fp128 %l4, %add1
  %add5 = fadd fp128 %l5, %add4
  store volatile fp128 %add0, fp128 *%ptr
  store volatile fp128 %add1, fp128 *%ptr
  store volatile fp128 %add4, fp128 *%ptr
  store volatile fp128 %add5, fp128 *%ptr
  ret void
}

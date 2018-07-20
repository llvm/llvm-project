; Test saving and restoring of call-saved FPRs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; This function should require all FPRs, but no other spill slots.
; We need to save and restore 8 of the 16 FPRs, so the frame size
; should be exactly 160 + 8 * 8 = 224.  The CFA offset is 160
; (the caller-allocated part of the frame) + 224.
define void @f1(float *%ptr) {
; CHECK-LABEL: f1:
; CHECK: aghi %r15, -224
; CHECK: .cfi_def_cfa_offset 384
; CHECK: std %f8, 216(%r15)
; CHECK: std %f9, 208(%r15)
; CHECK: std %f10, 200(%r15)
; CHECK: std %f11, 192(%r15)
; CHECK: std %f12, 184(%r15)
; CHECK: std %f13, 176(%r15)
; CHECK: std %f14, 168(%r15)
; CHECK: std %f15, 160(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f9, -176
; CHECK: .cfi_offset %f10, -184
; CHECK: .cfi_offset %f11, -192
; CHECK: .cfi_offset %f12, -200
; CHECK: .cfi_offset %f13, -208
; CHECK: .cfi_offset %f14, -216
; CHECK: .cfi_offset %f15, -224
; ...main function body...
; CHECK: ld %f8, 216(%r15)
; CHECK: ld %f9, 208(%r15)
; CHECK: ld %f10, 200(%r15)
; CHECK: ld %f11, 192(%r15)
; CHECK: ld %f12, 184(%r15)
; CHECK: ld %f13, 176(%r15)
; CHECK: ld %f14, 168(%r15)
; CHECK: ld %f15, 160(%r15)
; CHECK: aghi %r15, 224
; CHECK: br %r14
  %l0 = load volatile float, float *%ptr
  %l1 = load volatile float, float *%ptr
  %l2 = load volatile float, float *%ptr
  %l3 = load volatile float, float *%ptr
  %l4 = load volatile float, float *%ptr
  %l5 = load volatile float, float *%ptr
  %l6 = load volatile float, float *%ptr
  %l7 = load volatile float, float *%ptr
  %l8 = load volatile float, float *%ptr
  %l9 = load volatile float, float *%ptr
  %l10 = load volatile float, float *%ptr
  %l11 = load volatile float, float *%ptr
  %l12 = load volatile float, float *%ptr
  %l13 = load volatile float, float *%ptr
  %l14 = load volatile float, float *%ptr
  %l15 = load volatile float, float *%ptr
  %add0 = fadd float %l0, %l0
  %add1 = fadd float %l1, %add0
  %add2 = fadd float %l2, %add1
  %add3 = fadd float %l3, %add2
  %add4 = fadd float %l4, %add3
  %add5 = fadd float %l5, %add4
  %add6 = fadd float %l6, %add5
  %add7 = fadd float %l7, %add6
  %add8 = fadd float %l8, %add7
  %add9 = fadd float %l9, %add8
  %add10 = fadd float %l10, %add9
  %add11 = fadd float %l11, %add10
  %add12 = fadd float %l12, %add11
  %add13 = fadd float %l13, %add12
  %add14 = fadd float %l14, %add13
  %add15 = fadd float %l15, %add14
  store volatile float %add0, float *%ptr
  store volatile float %add1, float *%ptr
  store volatile float %add2, float *%ptr
  store volatile float %add3, float *%ptr
  store volatile float %add4, float *%ptr
  store volatile float %add5, float *%ptr
  store volatile float %add6, float *%ptr
  store volatile float %add7, float *%ptr
  store volatile float %add8, float *%ptr
  store volatile float %add9, float *%ptr
  store volatile float %add10, float *%ptr
  store volatile float %add11, float *%ptr
  store volatile float %add12, float *%ptr
  store volatile float %add13, float *%ptr
  store volatile float %add14, float *%ptr
  store volatile float %add15, float *%ptr
  ret void
}

; Like f1, but requires one fewer FPR.  We allocate in numerical order,
; so %f15 is the one that gets dropped.
define void @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aghi %r15, -216
; CHECK: .cfi_def_cfa_offset 376
; CHECK: std %f8, 208(%r15)
; CHECK: std %f9, 200(%r15)
; CHECK: std %f10, 192(%r15)
; CHECK: std %f11, 184(%r15)
; CHECK: std %f12, 176(%r15)
; CHECK: std %f13, 168(%r15)
; CHECK: std %f14, 160(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK: .cfi_offset %f9, -176
; CHECK: .cfi_offset %f10, -184
; CHECK: .cfi_offset %f11, -192
; CHECK: .cfi_offset %f12, -200
; CHECK: .cfi_offset %f13, -208
; CHECK: .cfi_offset %f14, -216
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 208(%r15)
; CHECK: ld %f9, 200(%r15)
; CHECK: ld %f10, 192(%r15)
; CHECK: ld %f11, 184(%r15)
; CHECK: ld %f12, 176(%r15)
; CHECK: ld %f13, 168(%r15)
; CHECK: ld %f14, 160(%r15)
; CHECK: aghi %r15, 216
; CHECK: br %r14
  %l0 = load volatile float, float *%ptr
  %l1 = load volatile float, float *%ptr
  %l2 = load volatile float, float *%ptr
  %l3 = load volatile float, float *%ptr
  %l4 = load volatile float, float *%ptr
  %l5 = load volatile float, float *%ptr
  %l6 = load volatile float, float *%ptr
  %l7 = load volatile float, float *%ptr
  %l8 = load volatile float, float *%ptr
  %l9 = load volatile float, float *%ptr
  %l10 = load volatile float, float *%ptr
  %l11 = load volatile float, float *%ptr
  %l12 = load volatile float, float *%ptr
  %l13 = load volatile float, float *%ptr
  %l14 = load volatile float, float *%ptr
  %add0 = fadd float %l0, %l0
  %add1 = fadd float %l1, %add0
  %add2 = fadd float %l2, %add1
  %add3 = fadd float %l3, %add2
  %add4 = fadd float %l4, %add3
  %add5 = fadd float %l5, %add4
  %add6 = fadd float %l6, %add5
  %add7 = fadd float %l7, %add6
  %add8 = fadd float %l8, %add7
  %add9 = fadd float %l9, %add8
  %add10 = fadd float %l10, %add9
  %add11 = fadd float %l11, %add10
  %add12 = fadd float %l12, %add11
  %add13 = fadd float %l13, %add12
  %add14 = fadd float %l14, %add13
  store volatile float %add0, float *%ptr
  store volatile float %add1, float *%ptr
  store volatile float %add2, float *%ptr
  store volatile float %add3, float *%ptr
  store volatile float %add4, float *%ptr
  store volatile float %add5, float *%ptr
  store volatile float %add6, float *%ptr
  store volatile float %add7, float *%ptr
  store volatile float %add8, float *%ptr
  store volatile float %add9, float *%ptr
  store volatile float %add10, float *%ptr
  store volatile float %add11, float *%ptr
  store volatile float %add12, float *%ptr
  store volatile float %add13, float *%ptr
  store volatile float %add14, float *%ptr
  ret void
}

; Like f1, but should require only one call-saved FPR.
define void @f3(float *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -168
; CHECK: .cfi_def_cfa_offset 328
; CHECK: std %f8, 160(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK-NOT: %f9
; CHECK-NOT: %f10
; CHECK-NOT: %f11
; CHECK-NOT: %f12
; CHECK-NOT: %f13
; CHECK-NOT: %f14
; CHECK-NOT: %f15
; ...main function body...
; CHECK: ld %f8, 160(%r15)
; CHECK: aghi %r15, 168
; CHECK: br %r14
  %l0 = load volatile float, float *%ptr
  %l1 = load volatile float, float *%ptr
  %l2 = load volatile float, float *%ptr
  %l3 = load volatile float, float *%ptr
  %l4 = load volatile float, float *%ptr
  %l5 = load volatile float, float *%ptr
  %l6 = load volatile float, float *%ptr
  %l7 = load volatile float, float *%ptr
  %l8 = load volatile float, float *%ptr
  %add0 = fadd float %l0, %l0
  %add1 = fadd float %l1, %add0
  %add2 = fadd float %l2, %add1
  %add3 = fadd float %l3, %add2
  %add4 = fadd float %l4, %add3
  %add5 = fadd float %l5, %add4
  %add6 = fadd float %l6, %add5
  %add7 = fadd float %l7, %add6
  %add8 = fadd float %l8, %add7
  store volatile float %add0, float *%ptr
  store volatile float %add1, float *%ptr
  store volatile float %add2, float *%ptr
  store volatile float %add3, float *%ptr
  store volatile float %add4, float *%ptr
  store volatile float %add5, float *%ptr
  store volatile float %add6, float *%ptr
  store volatile float %add7, float *%ptr
  store volatile float %add8, float *%ptr
  ret void
}

; This function should use all call-clobbered FPRs but no call-saved ones.
; It shouldn't need to create a frame.
define void @f4(float *%ptr) {
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
  %l0 = load volatile float, float *%ptr
  %l1 = load volatile float, float *%ptr
  %l2 = load volatile float, float *%ptr
  %l3 = load volatile float, float *%ptr
  %l4 = load volatile float, float *%ptr
  %l5 = load volatile float, float *%ptr
  %l6 = load volatile float, float *%ptr
  %l7 = load volatile float, float *%ptr
  %add0 = fadd float %l0, %l0
  %add1 = fadd float %l1, %add0
  %add2 = fadd float %l2, %add1
  %add3 = fadd float %l3, %add2
  %add4 = fadd float %l4, %add3
  %add5 = fadd float %l5, %add4
  %add6 = fadd float %l6, %add5
  %add7 = fadd float %l7, %add6
  store volatile float %add0, float *%ptr
  store volatile float %add1, float *%ptr
  store volatile float %add2, float *%ptr
  store volatile float %add3, float *%ptr
  store volatile float %add4, float *%ptr
  store volatile float %add5, float *%ptr
  store volatile float %add6, float *%ptr
  store volatile float %add7, float *%ptr
  ret void
}

; Like frame-03.ll, but for z13.  In this case we have 16 more registers
; available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

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
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f7, 0(%r2)
; CHECK-DAG: ld %f8, 0(%r2)
; CHECK-DAG: ld %f15, 0(%r2)
; CHECK-DAG: vlrepg %v16, 0(%r2)
; CHECK-DAG: vlrepg %v23, 0(%r2)
; CHECK-DAG: vlrepg %v24, 0(%r2)
; CHECK-DAG: vlrepg %v31, 0(%r2)
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
  %l16 = load volatile double, double *%ptr
  %l17 = load volatile double, double *%ptr
  %l18 = load volatile double, double *%ptr
  %l19 = load volatile double, double *%ptr
  %l20 = load volatile double, double *%ptr
  %l21 = load volatile double, double *%ptr
  %l22 = load volatile double, double *%ptr
  %l23 = load volatile double, double *%ptr
  %l24 = load volatile double, double *%ptr
  %l25 = load volatile double, double *%ptr
  %l26 = load volatile double, double *%ptr
  %l27 = load volatile double, double *%ptr
  %l28 = load volatile double, double *%ptr
  %l29 = load volatile double, double *%ptr
  %l30 = load volatile double, double *%ptr
  %l31 = load volatile double, double *%ptr
  %acc0 = fsub double %l0, %l0
  %acc1 = fsub double %l1, %acc0
  %acc2 = fsub double %l2, %acc1
  %acc3 = fsub double %l3, %acc2
  %acc4 = fsub double %l4, %acc3
  %acc5 = fsub double %l5, %acc4
  %acc6 = fsub double %l6, %acc5
  %acc7 = fsub double %l7, %acc6
  %acc8 = fsub double %l8, %acc7
  %acc9 = fsub double %l9, %acc8
  %acc10 = fsub double %l10, %acc9
  %acc11 = fsub double %l11, %acc10
  %acc12 = fsub double %l12, %acc11
  %acc13 = fsub double %l13, %acc12
  %acc14 = fsub double %l14, %acc13
  %acc15 = fsub double %l15, %acc14
  %acc16 = fsub double %l16, %acc15
  %acc17 = fsub double %l17, %acc16
  %acc18 = fsub double %l18, %acc17
  %acc19 = fsub double %l19, %acc18
  %acc20 = fsub double %l20, %acc19
  %acc21 = fsub double %l21, %acc20
  %acc22 = fsub double %l22, %acc21
  %acc23 = fsub double %l23, %acc22
  %acc24 = fsub double %l24, %acc23
  %acc25 = fsub double %l25, %acc24
  %acc26 = fsub double %l26, %acc25
  %acc27 = fsub double %l27, %acc26
  %acc28 = fsub double %l28, %acc27
  %acc29 = fsub double %l29, %acc28
  %acc30 = fsub double %l30, %acc29
  %acc31 = fsub double %l31, %acc30
  store volatile double %acc0, double *%ptr
  store volatile double %acc1, double *%ptr
  store volatile double %acc2, double *%ptr
  store volatile double %acc3, double *%ptr
  store volatile double %acc4, double *%ptr
  store volatile double %acc5, double *%ptr
  store volatile double %acc6, double *%ptr
  store volatile double %acc7, double *%ptr
  store volatile double %acc8, double *%ptr
  store volatile double %acc9, double *%ptr
  store volatile double %acc10, double *%ptr
  store volatile double %acc11, double *%ptr
  store volatile double %acc12, double *%ptr
  store volatile double %acc13, double *%ptr
  store volatile double %acc14, double *%ptr
  store volatile double %acc15, double *%ptr
  store volatile double %acc16, double *%ptr
  store volatile double %acc17, double *%ptr
  store volatile double %acc18, double *%ptr
  store volatile double %acc19, double *%ptr
  store volatile double %acc20, double *%ptr
  store volatile double %acc21, double *%ptr
  store volatile double %acc22, double *%ptr
  store volatile double %acc23, double *%ptr
  store volatile double %acc24, double *%ptr
  store volatile double %acc25, double *%ptr
  store volatile double %acc26, double *%ptr
  store volatile double %acc27, double *%ptr
  store volatile double %acc28, double *%ptr
  store volatile double %acc29, double *%ptr
  store volatile double %acc30, double *%ptr
  store volatile double %acc31, double *%ptr
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
; CHECK-NOT: %v15
; CHECK-NOT: %f15
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
  %l16 = load volatile double, double *%ptr
  %l17 = load volatile double, double *%ptr
  %l18 = load volatile double, double *%ptr
  %l19 = load volatile double, double *%ptr
  %l20 = load volatile double, double *%ptr
  %l21 = load volatile double, double *%ptr
  %l22 = load volatile double, double *%ptr
  %l23 = load volatile double, double *%ptr
  %l24 = load volatile double, double *%ptr
  %l25 = load volatile double, double *%ptr
  %l26 = load volatile double, double *%ptr
  %l27 = load volatile double, double *%ptr
  %l28 = load volatile double, double *%ptr
  %l29 = load volatile double, double *%ptr
  %l30 = load volatile double, double *%ptr
  %l31 = load volatile double, double *%ptr
  %acc0 = fsub double %l0, %l0
  %acc1 = fsub double %l1, %acc0
  %acc2 = fsub double %l2, %acc1
  %acc3 = fsub double %l3, %acc2
  %acc4 = fsub double %l4, %acc3
  %acc5 = fsub double %l5, %acc4
  %acc6 = fsub double %l6, %acc5
  %acc7 = fsub double %l7, %acc6
  %acc8 = fsub double %l8, %acc7
  %acc9 = fsub double %l9, %acc8
  %acc10 = fsub double %l10, %acc9
  %acc11 = fsub double %l11, %acc10
  %acc12 = fsub double %l12, %acc11
  %acc13 = fsub double %l13, %acc12
  %acc14 = fsub double %l14, %acc13
  %acc16 = fsub double %l16, %acc14
  %acc17 = fsub double %l17, %acc16
  %acc18 = fsub double %l18, %acc17
  %acc19 = fsub double %l19, %acc18
  %acc20 = fsub double %l20, %acc19
  %acc21 = fsub double %l21, %acc20
  %acc22 = fsub double %l22, %acc21
  %acc23 = fsub double %l23, %acc22
  %acc24 = fsub double %l24, %acc23
  %acc25 = fsub double %l25, %acc24
  %acc26 = fsub double %l26, %acc25
  %acc27 = fsub double %l27, %acc26
  %acc28 = fsub double %l28, %acc27
  %acc29 = fsub double %l29, %acc28
  %acc30 = fsub double %l30, %acc29
  %acc31 = fsub double %l31, %acc30
  store volatile double %acc0, double *%ptr
  store volatile double %acc1, double *%ptr
  store volatile double %acc2, double *%ptr
  store volatile double %acc3, double *%ptr
  store volatile double %acc4, double *%ptr
  store volatile double %acc5, double *%ptr
  store volatile double %acc6, double *%ptr
  store volatile double %acc7, double *%ptr
  store volatile double %acc8, double *%ptr
  store volatile double %acc9, double *%ptr
  store volatile double %acc10, double *%ptr
  store volatile double %acc11, double *%ptr
  store volatile double %acc12, double *%ptr
  store volatile double %acc13, double *%ptr
  store volatile double %acc14, double *%ptr
  store volatile double %acc16, double *%ptr
  store volatile double %acc17, double *%ptr
  store volatile double %acc18, double *%ptr
  store volatile double %acc19, double *%ptr
  store volatile double %acc20, double *%ptr
  store volatile double %acc21, double *%ptr
  store volatile double %acc22, double *%ptr
  store volatile double %acc23, double *%ptr
  store volatile double %acc24, double *%ptr
  store volatile double %acc25, double *%ptr
  store volatile double %acc26, double *%ptr
  store volatile double %acc27, double *%ptr
  store volatile double %acc28, double *%ptr
  store volatile double %acc29, double *%ptr
  store volatile double %acc30, double *%ptr
  store volatile double %acc31, double *%ptr
  ret void
}

; Like f1, but should require only one call-saved FPR.
define void @f3(double *%ptr) {
; CHECK-LABEL: f3:
; CHECK: aghi %r15, -8
; CHECK: .cfi_def_cfa_offset 168
; CHECK: std %f8, 0(%r15)
; CHECK: .cfi_offset %f8, -168
; CHECK-NOT: {{%[fv]9}}
; CHECK-NOT: {{%[fv]1[0-5]}}
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
  %l16 = load volatile double, double *%ptr
  %l17 = load volatile double, double *%ptr
  %l18 = load volatile double, double *%ptr
  %l19 = load volatile double, double *%ptr
  %l20 = load volatile double, double *%ptr
  %l21 = load volatile double, double *%ptr
  %l22 = load volatile double, double *%ptr
  %l23 = load volatile double, double *%ptr
  %l24 = load volatile double, double *%ptr
  %l25 = load volatile double, double *%ptr
  %l26 = load volatile double, double *%ptr
  %l27 = load volatile double, double *%ptr
  %l28 = load volatile double, double *%ptr
  %l29 = load volatile double, double *%ptr
  %l30 = load volatile double, double *%ptr
  %l31 = load volatile double, double *%ptr
  %acc0 = fsub double %l0, %l0
  %acc1 = fsub double %l1, %acc0
  %acc2 = fsub double %l2, %acc1
  %acc3 = fsub double %l3, %acc2
  %acc4 = fsub double %l4, %acc3
  %acc5 = fsub double %l5, %acc4
  %acc6 = fsub double %l6, %acc5
  %acc7 = fsub double %l7, %acc6
  %acc8 = fsub double %l8, %acc7
  %acc16 = fsub double %l16, %acc8
  %acc17 = fsub double %l17, %acc16
  %acc18 = fsub double %l18, %acc17
  %acc19 = fsub double %l19, %acc18
  %acc20 = fsub double %l20, %acc19
  %acc21 = fsub double %l21, %acc20
  %acc22 = fsub double %l22, %acc21
  %acc23 = fsub double %l23, %acc22
  %acc24 = fsub double %l24, %acc23
  %acc25 = fsub double %l25, %acc24
  %acc26 = fsub double %l26, %acc25
  %acc27 = fsub double %l27, %acc26
  %acc28 = fsub double %l28, %acc27
  %acc29 = fsub double %l29, %acc28
  %acc30 = fsub double %l30, %acc29
  %acc31 = fsub double %l31, %acc30
  store volatile double %acc0, double *%ptr
  store volatile double %acc1, double *%ptr
  store volatile double %acc2, double *%ptr
  store volatile double %acc3, double *%ptr
  store volatile double %acc4, double *%ptr
  store volatile double %acc5, double *%ptr
  store volatile double %acc6, double *%ptr
  store volatile double %acc7, double *%ptr
  store volatile double %acc8, double *%ptr
  store volatile double %acc16, double *%ptr
  store volatile double %acc17, double *%ptr
  store volatile double %acc18, double *%ptr
  store volatile double %acc19, double *%ptr
  store volatile double %acc20, double *%ptr
  store volatile double %acc21, double *%ptr
  store volatile double %acc22, double *%ptr
  store volatile double %acc23, double *%ptr
  store volatile double %acc24, double *%ptr
  store volatile double %acc25, double *%ptr
  store volatile double %acc26, double *%ptr
  store volatile double %acc27, double *%ptr
  store volatile double %acc28, double *%ptr
  store volatile double %acc29, double *%ptr
  store volatile double %acc30, double *%ptr
  store volatile double %acc31, double *%ptr
  ret void
}

; This function should use all call-clobbered FPRs and vector registers
; but no call-saved ones.  It shouldn't need to create a frame.
define void @f4(double *%ptr) {
; CHECK-LABEL: f4:
; CHECK-NOT: %r15
; CHECK-NOT: {{%[fv][89]}}
; CHECK-NOT: {{%[fv]1[0-5]}}
; CHECK: br %r14
  %l0 = load volatile double, double *%ptr
  %l1 = load volatile double, double *%ptr
  %l2 = load volatile double, double *%ptr
  %l3 = load volatile double, double *%ptr
  %l4 = load volatile double, double *%ptr
  %l5 = load volatile double, double *%ptr
  %l6 = load volatile double, double *%ptr
  %l7 = load volatile double, double *%ptr
  %l16 = load volatile double, double *%ptr
  %l17 = load volatile double, double *%ptr
  %l18 = load volatile double, double *%ptr
  %l19 = load volatile double, double *%ptr
  %l20 = load volatile double, double *%ptr
  %l21 = load volatile double, double *%ptr
  %l22 = load volatile double, double *%ptr
  %l23 = load volatile double, double *%ptr
  %l24 = load volatile double, double *%ptr
  %l25 = load volatile double, double *%ptr
  %l26 = load volatile double, double *%ptr
  %l27 = load volatile double, double *%ptr
  %l28 = load volatile double, double *%ptr
  %l29 = load volatile double, double *%ptr
  %l30 = load volatile double, double *%ptr
  %l31 = load volatile double, double *%ptr
  %acc0 = fsub double %l0, %l0
  %acc1 = fsub double %l1, %acc0
  %acc2 = fsub double %l2, %acc1
  %acc3 = fsub double %l3, %acc2
  %acc4 = fsub double %l4, %acc3
  %acc5 = fsub double %l5, %acc4
  %acc6 = fsub double %l6, %acc5
  %acc7 = fsub double %l7, %acc6
  %acc16 = fsub double %l16, %acc7
  %acc17 = fsub double %l17, %acc16
  %acc18 = fsub double %l18, %acc17
  %acc19 = fsub double %l19, %acc18
  %acc20 = fsub double %l20, %acc19
  %acc21 = fsub double %l21, %acc20
  %acc22 = fsub double %l22, %acc21
  %acc23 = fsub double %l23, %acc22
  %acc24 = fsub double %l24, %acc23
  %acc25 = fsub double %l25, %acc24
  %acc26 = fsub double %l26, %acc25
  %acc27 = fsub double %l27, %acc26
  %acc28 = fsub double %l28, %acc27
  %acc29 = fsub double %l29, %acc28
  %acc30 = fsub double %l30, %acc29
  %acc31 = fsub double %l31, %acc30
  store volatile double %acc0, double *%ptr
  store volatile double %acc1, double *%ptr
  store volatile double %acc2, double *%ptr
  store volatile double %acc3, double *%ptr
  store volatile double %acc4, double *%ptr
  store volatile double %acc5, double *%ptr
  store volatile double %acc6, double *%ptr
  store volatile double %acc7, double *%ptr
  store volatile double %acc16, double *%ptr
  store volatile double %acc17, double *%ptr
  store volatile double %acc18, double *%ptr
  store volatile double %acc19, double *%ptr
  store volatile double %acc20, double *%ptr
  store volatile double %acc21, double *%ptr
  store volatile double %acc22, double *%ptr
  store volatile double %acc23, double *%ptr
  store volatile double %acc24, double *%ptr
  store volatile double %acc25, double *%ptr
  store volatile double %acc26, double *%ptr
  store volatile double %acc27, double *%ptr
  store volatile double %acc28, double *%ptr
  store volatile double %acc29, double *%ptr
  store volatile double %acc30, double *%ptr
  store volatile double %acc31, double *%ptr
  ret void
}

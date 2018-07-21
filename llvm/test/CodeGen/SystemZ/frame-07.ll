; Test the saving and restoring of FPRs in large frames.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck -check-prefix=CHECK-NOFP %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -disable-fp-elim | FileCheck -check-prefix=CHECK-FP %s

; Test a frame size that requires some FPRs to be saved and loaded using
; the 20-bit STDY and LDY while others can use the 12-bit STD and LD.
; The frame is big enough to require two emergency spill slots at 160(%r15),
; as well as the 8 FPR save slots.  Get a frame of size 4128 by allocating
; (4128 - 176 - 8 * 8) / 8 = 486 extra doublewords.
define void @f1(double *%ptr, i64 %x) {
; CHECK-NOFP-LABEL: f1:
; CHECK-NOFP: aghi %r15, -4128
; CHECK-NOFP: .cfi_def_cfa_offset 4288
; CHECK-NOFP: stdy %f8, 4120(%r15)
; CHECK-NOFP: stdy %f9, 4112(%r15)
; CHECK-NOFP: stdy %f10, 4104(%r15)
; CHECK-NOFP: stdy %f11, 4096(%r15)
; CHECK-NOFP: std %f12, 4088(%r15)
; CHECK-NOFP: std %f13, 4080(%r15)
; CHECK-NOFP: std %f14, 4072(%r15)
; CHECK-NOFP: std %f15, 4064(%r15)
; CHECK-NOFP: .cfi_offset %f8, -168
; CHECK-NOFP: .cfi_offset %f9, -176
; CHECK-NOFP: .cfi_offset %f10, -184
; CHECK-NOFP: .cfi_offset %f11, -192
; CHECK-NOFP: .cfi_offset %f12, -200
; CHECK-NOFP: .cfi_offset %f13, -208
; CHECK-NOFP: .cfi_offset %f14, -216
; CHECK-NOFP: .cfi_offset %f15, -224
; ...main function body...
; CHECK-NOFP: ldy %f8, 4120(%r15)
; CHECK-NOFP: ldy %f9, 4112(%r15)
; CHECK-NOFP: ldy %f10, 4104(%r15)
; CHECK-NOFP: ldy %f11, 4096(%r15)
; CHECK-NOFP: ld %f12, 4088(%r15)
; CHECK-NOFP: ld %f13, 4080(%r15)
; CHECK-NOFP: ld %f14, 4072(%r15)
; CHECK-NOFP: ld %f15, 4064(%r15)
; CHECK-NOFP: aghi %r15, 4128
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f1:
; CHECK-FP: stmg %r11, %r15, 88(%r15)
; CHECK-FP: aghi %r15, -4128
; CHECK-FP: .cfi_def_cfa_offset 4288
; CHECK-FP: lgr %r11, %r15
; CHECK-FP: .cfi_def_cfa_register %r11
; CHECK-FP: stdy %f8, 4120(%r11)
; CHECK-FP: stdy %f9, 4112(%r11)
; CHECK-FP: stdy %f10, 4104(%r11)
; CHECK-FP: stdy %f11, 4096(%r11)
; CHECK-FP: std %f12, 4088(%r11)
; CHECK-FP: std %f13, 4080(%r11)
; CHECK-FP: std %f14, 4072(%r11)
; CHECK-FP: std %f15, 4064(%r11)
; ...main function body...
; CHECK-FP: ldy %f8, 4120(%r11)
; CHECK-FP: ldy %f9, 4112(%r11)
; CHECK-FP: ldy %f10, 4104(%r11)
; CHECK-FP: ldy %f11, 4096(%r11)
; CHECK-FP: ld %f12, 4088(%r11)
; CHECK-FP: ld %f13, 4080(%r11)
; CHECK-FP: ld %f14, 4072(%r11)
; CHECK-FP: ld %f15, 4064(%r11)
; CHECK-FP: lmg %r11, %r15, 4216(%r11)
; CHECK-FP: br %r14
  %y = alloca [486 x i64], align 8
  %elem = getelementptr inbounds [486 x i64], [486 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %elem
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

; Test a frame size that requires some FPRs to be saved and loaded using
; an indexed STD and LD while others can use the 20-bit STDY and LDY.
; The index can be any call-clobbered GPR except %r0.
;
; Don't require the accesses to share the same LLILH; that would be a
; good optimisation but is really a different test.
;
; As above, get a frame of size 524320 by allocating
; (524320 - 176 - 8 * 8) / 8 = 65510 extra doublewords.
define void @f2(double *%ptr, i64 %x) {
; CHECK-NOFP-LABEL: f2:
; CHECK-NOFP: agfi %r15, -524320
; CHECK-NOFP: .cfi_def_cfa_offset 524480
; CHECK-NOFP: llilh [[INDEX:%r[1-5]]], 8
; CHECK-NOFP: std %f8, 24([[INDEX]],%r15)
; CHECK-NOFP: std %f9, 16({{%r[1-5]}},%r15)
; CHECK-NOFP: std %f10, 8({{%r[1-5]}},%r15)
; CHECK-NOFP: std %f11, 0({{%r[1-5]}},%r15)
; CHECK-NOFP: stdy %f12, 524280(%r15)
; CHECK-NOFP: stdy %f13, 524272(%r15)
; CHECK-NOFP: stdy %f14, 524264(%r15)
; CHECK-NOFP: stdy %f15, 524256(%r15)
; CHECK-NOFP: .cfi_offset %f8, -168
; CHECK-NOFP: .cfi_offset %f9, -176
; CHECK-NOFP: .cfi_offset %f10, -184
; CHECK-NOFP: .cfi_offset %f11, -192
; CHECK-NOFP: .cfi_offset %f12, -200
; CHECK-NOFP: .cfi_offset %f13, -208
; CHECK-NOFP: .cfi_offset %f14, -216
; CHECK-NOFP: .cfi_offset %f15, -224
; ...main function body...
; CHECK-NOFP: ld %f8, 24({{%r[1-5]}},%r15)
; CHECK-NOFP: ld %f9, 16({{%r[1-5]}},%r15)
; CHECK-NOFP: ld %f10, 8({{%r[1-5]}},%r15)
; CHECK-NOFP: ld %f11, 0({{%r[1-5]}},%r15)
; CHECK-NOFP: ldy %f12, 524280(%r15)
; CHECK-NOFP: ldy %f13, 524272(%r15)
; CHECK-NOFP: ldy %f14, 524264(%r15)
; CHECK-NOFP: ldy %f15, 524256(%r15)
; CHECK-NOFP: agfi %r15, 524320
; CHECK-NOFP: br %r14
;
; CHECK-FP-LABEL: f2:
; CHECK-FP: stmg %r11, %r15, 88(%r15)
; CHECK-FP: agfi %r15, -524320
; CHECK-FP: .cfi_def_cfa_offset 524480
; CHECK-FP: llilh [[INDEX:%r[1-5]]], 8
; CHECK-FP: std %f8, 24([[INDEX]],%r11)
; CHECK-FP: std %f9, 16({{%r[1-5]}},%r11)
; CHECK-FP: std %f10, 8({{%r[1-5]}},%r11)
; CHECK-FP: std %f11, 0({{%r[1-5]}},%r11)
; CHECK-FP: stdy %f12, 524280(%r11)
; CHECK-FP: stdy %f13, 524272(%r11)
; CHECK-FP: stdy %f14, 524264(%r11)
; CHECK-FP: stdy %f15, 524256(%r11)
; CHECK-FP: .cfi_offset %f8, -168
; CHECK-FP: .cfi_offset %f9, -176
; CHECK-FP: .cfi_offset %f10, -184
; CHECK-FP: .cfi_offset %f11, -192
; CHECK-FP: .cfi_offset %f12, -200
; CHECK-FP: .cfi_offset %f13, -208
; CHECK-FP: .cfi_offset %f14, -216
; CHECK-FP: .cfi_offset %f15, -224
; ...main function body...
; CHECK-FP: ld %f8, 24({{%r[1-5]}},%r11)
; CHECK-FP: ld %f9, 16({{%r[1-5]}},%r11)
; CHECK-FP: ld %f10, 8({{%r[1-5]}},%r11)
; CHECK-FP: ld %f11, 0({{%r[1-5]}},%r11)
; CHECK-FP: ldy %f12, 524280(%r11)
; CHECK-FP: ldy %f13, 524272(%r11)
; CHECK-FP: ldy %f14, 524264(%r11)
; CHECK-FP: ldy %f15, 524256(%r11)
; CHECK-FP: aghi %r11, 128
; CHECK-FP: lmg %r11, %r15, 524280(%r11)
; CHECK-FP: br %r14
  %y = alloca [65510 x i64], align 8
  %elem = getelementptr inbounds [65510 x i64], [65510 x i64]* %y, i64 0, i64 0
  store volatile i64 %x, i64* %elem
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

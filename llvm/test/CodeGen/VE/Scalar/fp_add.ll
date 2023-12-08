; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define float @func1(float %a, float %b) {
; CHECK-LABEL: func1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fadd.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd float %a, %b
  ret float %r
}

define double @func2(double %a, double %b) {
; CHECK-LABEL: func2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd double %a, %b
  ret double %r
}

define fp128 @func3(fp128 %a, fp128 %b) {
; CHECK-LABEL: func3:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fadd.q %s0, %s0, %s2
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd fp128 %a, %b
  ret fp128 %r
}

define float @func4(float %a) {
; CHECK-LABEL: func4:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 1084227584
; CHECK-NEXT:    fadd.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd float %a, 5.000000e+00
  ret float %r
}

define double @func5(double %a) {
; CHECK-LABEL: func5:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 1075052544
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd double %a, 5.000000e+00
  ret double %r
}

define fp128 @func6(fp128 %a) {
; CHECK-LABEL: func6:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fadd.q %s0, %s0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd fp128 %a, 0xL00000000000000004001400000000000
  ret fp128 %r
}

define float @func7(float %a) {
; CHECK-LABEL: func7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea.sl %s1, 2139095039
; CHECK-NEXT:    fadd.s %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd float %a, 0x47EFFFFFE0000000
  ret float %r
}

define double @func8(double %a) {
; CHECK-LABEL: func8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, -1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, 2146435071(, %s1)
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd double %a, 0x7FEFFFFFFFFFFFFF
  ret double %r
}

define fp128 @func9(fp128 %a) {
; CHECK-LABEL: func9:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fadd.q %s0, %s0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd fp128 %a, 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF
  ret fp128 %r
}

define float @fadds_imm(float %a) {
; CHECK-LABEL: fadds_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fadd.s %s0, %s0, (2)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd float %a, -2.e+00
  ret float %r
}

define double @faddd_imm(double %a) {
; CHECK-LABEL: faddd_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fadd.d %s0, %s0, (2)1
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd double %a, -2.e+00
  ret double %r
}

define fp128 @faddq_imm(fp128 %a) {
; CHECK-LABEL: faddq_imm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fadd.q %s0, %s0, %s4
; CHECK-NEXT:    b.l.t (, %s10)
  %r = fadd fp128 %a, 0xLA0000000000000000000000000000000
  ret fp128 %r
}

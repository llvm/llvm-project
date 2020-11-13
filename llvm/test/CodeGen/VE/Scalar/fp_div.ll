; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define float @func1(float %a, float %b) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fdiv.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv float %a, %b
  ret float %r
}

define double @func2(double %a, double %b) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fdiv.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv double %a, %b
  ret double %r
}

define fp128 @func3(fp128 %a, fp128 %b) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s4, __divtf3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __divtf3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv fp128 %a, %b
  ret fp128 %r
}

define float @func4(float %a) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 1084227584
; CHECK-NEXT:    fdiv.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv float %a, 5.000000e+00
  ret float %r
}

define double @func5(double %a) {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 1075052544
; CHECK-NEXT:    fdiv.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv double %a, 5.000000e+00
  ret double %r
}

define fp128 @func6(fp128 %a) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s4)
; CHECK-NEXT:    ld %s3, (, %s4)
; CHECK-NEXT:    lea %s4, __divtf3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __divtf3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv fp128 %a, 0xL00000000000000004001400000000000
  ret fp128 %r
}

define float @func7(float %a) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, 2139095039
; CHECK-NEXT:    fdiv.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv float %a, 0x47EFFFFFE0000000
  ret float %r
}

define double @func8(double %a) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, -1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, 2146435071(, %s1)
; CHECK-NEXT:    fdiv.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv double %a, 0x7FEFFFFFFFFFFFFF
  ret double %r
}

; Function Attrs: norecurse nounwind readnone
define fp128 @func9(fp128 %a) {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s4, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s2, 8(, %s4)
; CHECK-NEXT:    ld %s3, (, %s4)
; CHECK-NEXT:    lea %s4, __divtf3@lo
; CHECK-NEXT:    and %s4, %s4, (32)0
; CHECK-NEXT:    lea.sl %s12, __divtf3@hi(, %s4)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fdiv fp128 %a, 0xLFFFFFFFFFFFFFFFF7FFEFFFFFFFFFFFF
  ret fp128 %r
}

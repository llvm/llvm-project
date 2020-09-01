; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

define float @func1(float %a, float %b) {
; CHECK-LABEL: func1:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fsub.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fsub float %a, %b
  ret float %r
}

define double @func2(double %a, double %b) {
; CHECK-LABEL: func2:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fsub.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fsub double %a, %b
  ret double %r
}

define fp128 @func3(fp128 %a, fp128 %b) {
; CHECK-LABEL: func3:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fsub.q %s0, %s0, %s2
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fsub fp128 %a, %b
  ret fp128 %r
}

define float @func4(float %a) {
; CHECK-LABEL: func4:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, -1063256064
; CHECK-NEXT:    fadd.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd float %a, -5.000000e+00
  ret float %r
}

define double @func5(double %a) {
; CHECK-LABEL: func5:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, -1072431104
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd double %a, -5.000000e+00
  ret double %r
}

define fp128 @func6(fp128 %a) {
; CHECK-LABEL: func6:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fadd.q %s0, %s0, %s4
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd fp128 %a, 0xL0000000000000000C001400000000000
  ret fp128 %r
}

define float @func7(float %a) {
; CHECK-LABEL: func7:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea.sl %s1, -8388609
; CHECK-NEXT:    fadd.s %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd float %a, 0xC7EFFFFFE0000000
  ret float %r
}

define double @func8(double %a) {
; CHECK-LABEL: func8:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s1, -1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, -1048577(, %s1)
; CHECK-NEXT:    fadd.d %s0, %s0, %s1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd double %a, 0xFFEFFFFFFFFFFFFF
  ret double %r
}

define fp128 @func9(fp128 %a) {
; CHECK-LABEL: func9:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fadd.q %s0, %s0, %s4
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fadd fp128 %a, 0xLFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFF
  ret fp128 %r
}

define float @fsubs_ir(float %a) {
; CHECK-LABEL: fsubs_ir:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fsub.s %s0, 0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fsub float 0.e+00, %a
  ret float %r
}

define float @fsubs_ri(float %a) {
; CHECK-LABEL: fsubs_ri:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fadd.s %s0, %s0, (2)1
; CHECK-NEXT:    or %s11, 0, %s9
  %r = fsub float %a, 2.0e+00
  ret float %r
}

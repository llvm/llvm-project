; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

; rdar://9332258

define float @test1(float %x, float %y) nounwind {
entry:
; CHECK-LABEL: test1:
; CHECK: movi.4s	v2, #128, lsl #24
; CHECK: bit.16b	v0, v1, v2
  %0 = tail call float @copysignf(float %x, float %y) nounwind readnone
  ret float %0
}

define double @test2(double %x, double %y) nounwind {
entry:
; CHECK-LABEL: test2:
; CHECK: movi.2d	v2, #0
; CHECK: fneg.2d	v2, v2
; CHECK: bit.16b	v0, v1, v2
  %0 = tail call double @copysign(double %x, double %y) nounwind readnone
  ret double %0
}

; rdar://9545768
define double @test3(double %a, float %b, float %c) nounwind {
; CHECK-LABEL: test3:
; CHECK: fcvt d1, s1
; CHECK: fneg.2d v2, v{{[0-9]+}}
; CHECK: bit.16b v0, v1, v2
  %tmp1 = fadd float %b, %c
  %tmp2 = fpext float %tmp1 to double
  %tmp = tail call double @copysign( double %a, double %tmp2 ) nounwind readnone
  ret double %tmp
}

define float @test4() nounwind {
entry:
; CHECK-LABEL: test4:
; CHECK: fcvt s0, d0
; CHECK: movi.4s v[[CONST:[0-9]+]], #128, lsl #24
; CHECK: bit.16b v{{[0-9]+}}, v0, v[[CONST]]
  %0 = tail call double (...) @bar() nounwind
  %1 = fptrunc double %0 to float
  %2 = tail call float @copysignf(float 5.000000e-01, float %1) nounwind readnone
  %3 = fadd float %1, %2
  ret float %3
}

declare double @bar(...)
declare double @copysign(double, double) nounwind readnone
declare float @copysignf(float, float) nounwind readnone

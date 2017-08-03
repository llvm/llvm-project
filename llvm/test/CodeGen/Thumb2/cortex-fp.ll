; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -march=thumb -mcpu=cortex-m3 | FileCheck %s -check-prefix=CHECK -check-prefix=CORTEXM3
; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -march=thumb -mcpu=cortex-m4 | FileCheck %s -check-prefix=CHECK -check-prefix=CORTEXM4
; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -march=thumb -mcpu=cortex-m7 | FileCheck %s -check-prefix=CHECK -check-prefix=CORTEXM7
; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 -march=thumb -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK -check-prefix=CORTEXA8


define float @foo(float %a, float %b) {
entry:
; CHECK-LABEL: foo:
; CORTEXM3: bl ___mulsf3
; CORTEXM4: vmul.f32  s
; CORTEXM7: vmul.f32  s
; CORTEXA8: vmul.f32  d
  %0 = fmul float %a, %b
  ret float %0
}

define double @bar(double %a, double %b) {
entry:
; CHECK-LABEL: bar:
  %0 = fmul double %a, %b
; CORTEXM3: bl ___muldf3
; CORTEXM4: {{bl|b.w}} ___muldf3
; CORTEXM7: vmul.f64  d
; CORTEXA8: vmul.f64  d
  ret double %0
}

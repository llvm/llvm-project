; RUN: llc < %s -mtriple=armv7-apple-ios -mattr=+vfp2 | FileCheck %s

define fastcc double @t1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %a, float %b) {
entry:
; CHECK-LABEL: t1:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd float %a, %b
  %conv = fpext float %add to double
  ret double %conv
}

define fastcc double @t2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %a, float %b, double %c) {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd double %a, %c
  ret double %add
}

define fastcc float @t3(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, float %a, double %b, float %c) {
entry:
; CHECK-LABEL: t3:
; CHECK: vldr
  %add = fadd float %a, %c
  ret float %add
}

define fastcc double @t4(double %a, double %b) #0 {
entry:
; CHECK-LABEL: t4:
; CHECK: vstr
  %add = fadd double %a, %b
  %sub = fsub double %a, %b
  %call = tail call fastcc double @x(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double %add, float 0.000000e+00, double %sub) #2
  ret double %call
}

declare fastcc double @x(double, double, double, double, double, double, double, float, double)

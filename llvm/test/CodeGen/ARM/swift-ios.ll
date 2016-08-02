; RUN: llc -mtriple=armv7-apple-ios < %s | FileCheck %s

define swiftcc float @t1(float %a, float %b) {
entry:
; CHECK: t1
; CHECK-NOT: vmov
; CHECK: vadd.f32
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  store float %b, float* %b.addr, align 4
  %0 = load float, float* %a.addr, align 4
  %1 = load float, float* %b.addr, align 4
  %add = fadd float %0, %1
  ret float %add
}

define swiftcc double @t2(double %a, double %b) {
entry:
; CHECK: t2
; CHECK-NOT: vmov
; CHECK: vadd.f64
  %a.addr = alloca double, align 8
  %b.addr = alloca double, align 8
  store double %a, double* %a.addr, align 8
  store double %b, double* %b.addr, align 8
  %0 = load double, double* %a.addr, align 8
  %1 = load double, double* %b.addr, align 8
  %add = fadd double %0, %1
  ret double %add
}

define swiftcc double @t9(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %a, float %b) {
entry:
; CHECK-LABEL: t9:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd float %a, %b
  %conv = fpext float %add to double
  ret double %conv
}

define swiftcc double @t10(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %a, float %b, double %c) {
entry:
; CHECK-LABEL: t10:
; CHECK-NOT: vmov
; CHECK: vldr
  %add = fadd double %a, %c
  ret double %add
}

define swiftcc float @t11(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, float %a, double %b, float %c) {
entry:
; CHECK-LABEL: t11:
; CHECK: vldr
  %add = fadd float %a, %c
  ret float %add
}

define swiftcc double @t12(double %a, double %b) {
entry:
; CHECK-LABEL: t12:
; CHECK: vstr
  %add = fadd double %a, %b
  %sub = fsub double %a, %b
  %call = tail call swiftcc double @x(double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00, double %add, float 0.000000e+00, double %sub)
  ret double %call
}

declare swiftcc double @x(double, double, double, double, double, double, double, float, double)

attributes #0 = { readnone }
attributes #1 = { readonly }

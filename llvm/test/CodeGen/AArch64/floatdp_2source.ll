; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu -mcpu=cyclone | FileCheck %s

@varfloat = global float 0.0
@vardouble = global double 0.0

define void @testfloat() {
; CHECK-LABEL: testfloat:
  %val1 = load float, ptr @varfloat

  %val2 = fadd float %val1, %val1
; CHECK: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}

  %val3 = fmul float %val2, %val1
; CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}

  %val4 = fdiv float %val3, %val1
; CHECK: fdiv {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}

  %val5 = fsub float %val4, %val2
; CHECK: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}

  store volatile float %val5, ptr @varfloat

; These will be enabled with the implementation of floating-point litpool entries.
  %val6 = fmul float %val1, %val2
  %val7 = fsub float -0.0, %val6
; CHECK: fnmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}

  store volatile float %val7, ptr @varfloat

  ret void
}

define void @testdouble() {
; CHECK-LABEL: testdouble:
  %val1 = load double, ptr @vardouble

  %val2 = fadd double %val1, %val1
; CHECK: fadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

  %val3 = fmul double %val2, %val1
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

  %val4 = fdiv double %val3, %val1
; CHECK: fdiv {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

  %val5 = fsub double %val4, %val2
; CHECK: fsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

  store volatile double %val5, ptr @vardouble

; These will be enabled with the implementation of doubleing-point litpool entries.
   %val6 = fmul double %val1, %val2
   %val7 = fsub double -0.0, %val6
; CHECK: fnmul {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}

   store volatile double %val7, ptr @vardouble

  ret void
}

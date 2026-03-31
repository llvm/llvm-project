; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu -mcpu=cyclone | FileCheck %s

@varfloat = global float 0.0
@vardouble = global double 0.0

; llvm.fadd/fsub/fmul intrinsics lower to the same instructions as plain
; fadd/fsub/fmul — the fp.control operand bundle only affects backends that
; support per-instruction FTZ (e.g. NVPTX), not AArch64.

define float @fadd_f_intrinsic(float %a, float %b) {
; CHECK-LABEL: fadd_f_intrinsic:
; CHECK: fadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %r = call float @llvm.fadd.f32(float %a, float %b)
  ret float %r
}

define float @fsub_f_intrinsic(float %a, float %b) {
; CHECK-LABEL: fsub_f_intrinsic:
; CHECK: fsub {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %r = call float @llvm.fsub.f32(float %a, float %b)
  ret float %r
}

define float @fmul_f_intrinsic(float %a, float %b) {
; CHECK-LABEL: fmul_f_intrinsic:
; CHECK: fmul {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %r = call float @llvm.fmul.f32(float %a, float %b)
  ret float %r
}

define double @fadd_d_intrinsic(double %a, double %b) {
; CHECK-LABEL: fadd_d_intrinsic:
; CHECK: fadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

define double @fsub_d_intrinsic(double %a, double %b) {
; CHECK-LABEL: fsub_d_intrinsic:
; CHECK: fsub {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

define double @fmul_d_intrinsic(double %a, double %b) {
; CHECK-LABEL: fmul_d_intrinsic:
; CHECK: fmul {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %r = call double @llvm.fmul.f64(double %a, double %b)
  ret double %r
}

declare float @llvm.fadd.f32(float, float)
declare float @llvm.fsub.f32(float, float)
declare float @llvm.fmul.f32(float, float)
declare double @llvm.fadd.f64(double, double)
declare double @llvm.fsub.f64(double, double)
declare double @llvm.fmul.f64(double, double)
declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)

define float @fma_f_intrinsic(float %a, float %b, float %c) {
; CHECK-LABEL: fma_f_intrinsic:
; CHECK: fmadd {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}, {{s[0-9]+}}
  %r = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define double @fma_d_intrinsic(double %a, double %b, double %c) {
; CHECK-LABEL: fma_d_intrinsic:
; CHECK: fmadd {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}
  %r = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %r
}

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

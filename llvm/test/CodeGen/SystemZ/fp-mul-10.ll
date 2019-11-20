; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.fma.f64(double %f1, double %f2, double %f3)
declare float @llvm.fma.f32(float %f1, float %f2, float %f3)

define double @f1(double %f1, double %f2, double %acc) {
; CHECK-LABEL: f1:
; CHECK: wfnmadb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call double @llvm.fma.f64 (double %f1, double %f2, double %acc)
  %negres = fneg double %res
  ret double %negres
}

define double @f2(double %f1, double %f2, double %acc) {
; CHECK-LABEL: f2:
; CHECK: wfnmsdb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %negacc = fneg double %acc
  %res = call double @llvm.fma.f64 (double %f1, double %f2, double %negacc)
  %negres = fneg double %res
  ret double %negres
}

define float @f3(float %f1, float %f2, float %acc) {
; CHECK-LABEL: f3:
; CHECK: wfnmasb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %acc)
  %negres = fneg float %res
  ret float %negres
}

define float @f4(float %f1, float %f2, float %acc) {
; CHECK-LABEL: f4:
; CHECK: wfnmssb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %negacc = fneg float %acc
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %negacc)
  %negres = fneg float %res
  ret float %negres
}


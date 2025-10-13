; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.fma.f64(double %f1, double %f2, double %f3)
declare float @llvm.fma.f32(float %f1, float %f2, float %f3)
declare half @llvm.fma.f16(half %f1, half %f2, half %f3)

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

define half @f3_half(half %f1, half %f2, half %acc) {
; CHECK-LABEL: f3_half:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: wfmasb %f0, %f0, %f8, %f10
; CHECK: brasl %r14, __truncsfhf2@PLT
; CHECK-NOT: brasl
; CHECK:      lcdfr %f0, %f0
; CHECK-NEXT: lmg
; CHECK-NEXT: br %r14
  %res = call half @llvm.fma.f16 (half %f1, half %f2, half %acc)
  %negres = fneg half %res
  ret half %negres
}

define float @f3(float %f1, float %f2, float %acc) {
; CHECK-LABEL: f3:
; CHECK: wfnmasb %f0, %f0, %f2, %f4
; CHECK: br %r14
  %res = call float @llvm.fma.f32 (float %f1, float %f2, float %acc)
  %negres = fneg float %res
  ret float %negres
}

define half @f4_half(half %f1, half %f2, half %acc) {
; CHECK-LABEL: f4_half:
; CHECK-NOT: brasl
; CHECK: lcdfr %f0, %f4
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: wfmasb %f0, %f0, %f8, %f10
; CHECK: brasl %r14, __truncsfhf2@PLT
; CHECK-NOT: brasl
; CHECK:      lcdfr %f0, %f0
; CHECK-NEXT: lmg
; CHECK-NEXT: br %r14
  %negacc = fneg half %acc
  %res = call half @llvm.fma.f16 (half %f1, half %f2, half %negacc)
  %negres = fneg half %res
  ret half %negres
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

; Test copysign intrinsics involving half.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare half @llvm.copysign.f16(half, half)
declare float @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)

; Test f16 copies.
define half @f0(half %a, half %b) {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: cpsdr %f0, %f9, %f0
; CHECK: brasl %r14, __truncsfhf2@PLT
; CHECK: br %r14
  %res = call half @llvm.copysign.f16(half %a, half %b)
  ret half %res
}

; Test f16 copies where the sign comes from an f32.
define half @f1(half %a, float %b) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: cpsdr %f0, %f8, %f0
; CHECK: brasl %r14, __truncsfhf2@PLT
; CHECK: br %r14
  %bh = fptrunc float %b to half
  %res = call half @llvm.copysign.f16(half %a, half %bh)
  ret half %res
}

; Test f16 copies where the sign comes from an f64.
define half @f2(half %a, double %b) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __extendhfdf2@PLT
; CHECK: cpsdr %f0, %f8, %f0
; CHECK: brasl %r14, __truncdfhf2@PLT
; CHECK: br %r14
  %bh = fptrunc double %b to half
  %res = call half @llvm.copysign.f16(half %a, half %bh)
  ret half %res
}

; Test f32 copies in which the sign comes from an f16.
define float @f3(float %a, half %b) {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK: cpsdr %f0, %f0, %f8
; CHECK: br %r14
  %bf = fpext half %b to float
  %res = call float @llvm.copysign.f32(float %a, float %bf)
  ret float %res
}

; Test f64 copies in which the sign comes from an f16.
define double @f4(double %a, half %b) {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, __extendhfdf2@PLT
; CHECK: cpsdr %f0, %f0, %f8
; CHECK: br %r14
  %bd = fpext half %b to double
  %res = call double @llvm.copysign.f64(double %a, double %bd)
  ret double %res
}

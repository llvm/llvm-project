; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s

declare double @llvm.powi.f64.i32(double, i32)
declare float @llvm.powi.f32.i32(float, i32)

define arm_aapcs_vfpcc double @d(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64.i32(double %d, i32 %i)
  ret double %0
}

; CHECK-LABEL: d:
; CHECK: vmov s[[REGISTER:[0-9]+]], r0
; CHECK-NEXT: vcvt.f64.s32 d1, s[[REGISTER]]
; CHECK-NEXT: b.w pow
; CHECK-NOT: __powisf2

define arm_aapcs_vfpcc float @f(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32.i32(float %f, i32 %i)
  ret float %0
}

; CHECK-LABEL: f:
; CHECK: vmov s[[REGISTER:[0-9]+]], r0
; CHECK-NEXT: vcvt.f32.s32 s1, s[[REGISTER]]
; CHECK-NEXT: b.w pow
; CHECK-NOT: __powisf2

define arm_aapcs_vfpcc float @g(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64.i32(double %d, i32 %i)
  %conv = fptrunc double %0 to float
  ret float %conv
}

; CHECK-LABEL: g:
; CHECK: vmov s[[REGISTER:[0-9]+]], r0
; CHECK-NEXT: vcvt.f64.s32 d1, s[[REGISTER]]
; CHECK-NEXT: bl pow
; CHECK-NOT: bl __powidf2
; CHECK-NEXT: vcvt.f32.f64 s0, d0

define arm_aapcs_vfpcc double @h(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32.i32(float %f, i32 %i)
  %conv = fpext float %0 to double
  ret double %conv
}

; CHECK-LABEL: h:
; CHECK: vmov s[[REGISTER:[0-9]+]], r0
; CHECK-NEXT: vcvt.f32.s32 s1, s[[REGISTER]]
; CHECK-NEXT: bl powf
; CHECK-NOT: bl __powisf2
; CHECK-NEXT: vcvt.f64.f32 d0, s0


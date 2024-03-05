; RUN: llc < %s -mtriple=aarch64 -mattr=+v8.5a  | FileCheck %s

declare float @llvm.aarch64.frint32z.f32(float)
declare double @llvm.aarch64.frint32z.f64(double)
declare float @llvm.aarch64.frint64z.f32(float)
declare double @llvm.aarch64.frint64z.f64(double)

define dso_local float @t_frint32z(float %a) {
; CHECK-LABEL: t_frint32z:
; CHECK:         frint32z s0, s0
; CHECK-NEXT:    ret
entry:
  %val = tail call float @llvm.aarch64.frint32z.f32(float %a)
  ret float %val
}

define dso_local double @t_frint32zf(double %a) {
; CHECK-LABEL: t_frint32zf:
; CHECK:         frint32z d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call double @llvm.aarch64.frint32z.f64(double %a)
  ret double %val
}

define dso_local float @t_frint64z(float %a) {
; CHECK-LABEL: t_frint64z:
; CHECK:         frint64z s0, s0
; CHECK-NEXT:    ret
entry:
  %val = tail call float @llvm.aarch64.frint64z.f32(float %a)
  ret float %val
}

define dso_local double @t_frint64zf(double %a) {
; CHECK-LABEL: t_frint64zf:
; CHECK:         frint64z d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call double @llvm.aarch64.frint64z.f64(double %a)
  ret double %val
}

declare float @llvm.aarch64.frint32x.f32(float)
declare double @llvm.aarch64.frint32x.f64(double)
declare float @llvm.aarch64.frint64x.f32(float)
declare double @llvm.aarch64.frint64x.f64(double)

define dso_local float @t_frint32x(float %a) {
; CHECK-LABEL: t_frint32x:
; CHECK:         frint32x s0, s0
; CHECK-NEXT:    ret
entry:
  %val = tail call float @llvm.aarch64.frint32x.f32(float %a)
  ret float %val
}

define dso_local double @t_frint32xf(double %a) {
; CHECK-LABEL: t_frint32xf:
; CHECK:         frint32x d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call double @llvm.aarch64.frint32x.f64(double %a)
  ret double %val
}

define dso_local float @t_frint64x(float %a) {
; CHECK-LABEL: t_frint64x:
; CHECK:         frint64x s0, s0
; CHECK-NEXT:    ret
entry:
  %val = tail call float @llvm.aarch64.frint64x.f32(float %a)
  ret float %val
}

define dso_local double @t_frint64xf(double %a) {
; CHECK-LABEL: t_frint64xf:
; CHECK:         frint64x d0, d0
; CHECK-NEXT:    ret
entry:
  %val = tail call double @llvm.aarch64.frint64x.f64(double %a)
  ret double %val
}

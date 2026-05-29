; AIR's air.fmin/fmax drop NaN; wrap renamed calls in a NaN-propagation select.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK-NOT: llvm.minimum
; CHECK-NOT: llvm.maximum
; CHECK: call float @air.fmin.f32
; CHECK: fcmp uno
; CHECK: select i1
; CHECK: call float @air.fmax.f32
; CHECK: fcmp uno
; CHECK: select i1
; CHECK: ret void

declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)

define void @kernel(float %a, float %b, ptr %out) {
entry:
  %mn = call float @llvm.minimum.f32(float %a, float %b)
  %mx = call float @llvm.maximum.f32(float %a, float %b)
  %s = fadd float %mn, %mx
  store float %s, ptr %out
  ret void
}

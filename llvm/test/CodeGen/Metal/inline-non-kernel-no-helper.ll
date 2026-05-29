; Negative test: a kernel with no internal callees is not modified by the
; inline-non-kernel pass; no spurious inlining of intrinsic declarations.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: call float @air.fmin.f32
; CHECK: ret void

declare float @air.fmin.f32(float, float)

define void @kernel(float %a, float %b, ptr %out) {
entry:
  %m = call float @air.fmin.f32(float %a, float %b)
  store float %m, ptr %out
  ret void
}

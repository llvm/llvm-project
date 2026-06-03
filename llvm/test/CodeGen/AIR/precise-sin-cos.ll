; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-NOT: air.fast_sin
; CHECK-NOT: air.fast_cos
; CHECK-LABEL: define void @kernel
; CHECK: call float @air.sin.f32(float
; CHECK: call float @air.cos.f32(float

define void @kernel(ptr addrspace(1) %out, float %x) {
entry:
  %s = call float @llvm.sin.f32(float %x)
  %c = call float @llvm.cos.f32(float %x)
  %r = fadd float %s, %c
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)

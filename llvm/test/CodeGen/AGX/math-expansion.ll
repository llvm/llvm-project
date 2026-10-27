; The math functions AGX has no instruction for are built from the ones it
; has (AGXMathExpansion.cpp), in float for half: no library calls remain.
; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-codegenprepare %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: @f(
; CHECK-NOT: call float @llvm.{{tan|asin|acos|atan|atan2|sinh|cosh|tanh}}.f32
; CHECK: call float @llvm.sin.f32
; CHECK: call float @llvm.cos.f32
; CHECK: call float @llvm.exp.f32
; CHECK-LABEL: @h(
; CHECK: fpext half
; CHECK: fptrunc float {{.*}} to half
; CHECK-NOT: call half @llvm.atan.f16
; ASM-LABEL: f:
; ASM-NOT: {{tanf|atanf|call}}

define agx_kernel void @f(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %a = call float @llvm.tan.f32(float %x)
  %b = call float @llvm.asin.f32(float %a)
  %c = call float @llvm.acos.f32(float %b)
  %d = call float @llvm.atan.f32(float %c)
  %e = call float @llvm.atan2.f32(float %d, float %x)
  %f = call float @llvm.sinh.f32(float %e)
  %g = call float @llvm.cosh.f32(float %f)
  %h = call float @llvm.tanh.f32(float %g)
  store float %h, ptr addrspace(1) %p
  ret void
}

define agx_kernel void @h(ptr addrspace(1) %p) {
  %x = load half, ptr addrspace(1) %p
  %a = call half @llvm.atan.f16(half %x)
  store half %a, ptr addrspace(1) %p
  ret void
}

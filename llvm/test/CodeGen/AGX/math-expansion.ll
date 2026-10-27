; The math functions AGX has no instruction for are built from the ones it
; has (AGXMathExpansion.cpp), in float for half: no library calls remain.
; Precise float sin, cos, exp and log are built too (exact reduction and
; polynomials), with the hardware's sin and cos kept for huge arguments; the
; afn forms stay the hardware's.
; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-codegenprepare %s -o - | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o - | FileCheck %s --check-prefix=ASM

; CHECK-LABEL: @f(
; CHECK-NOT: call float @llvm.{{tan|asin|acos|atan|atan2|sinh|cosh|tanh|sin|cos|exp|log}}.f32
; CHECK: call float @llvm.roundeven.f32
; CHECK: call afn float @llvm.sin.f32
; CHECK: call afn float @llvm.cos.f32
; CHECK-NOT: call float @llvm.{{tan|asin|acos|atan|atan2|sinh|cosh|tanh|sin|cos|exp|log}}.f32
; CHECK-LABEL: @h(
; CHECK: fpext half
; CHECK: fptrunc float {{.*}} to half
; CHECK-NOT: call half @llvm.atan.f16
; CHECK-LABEL: @sc(
; CHECK-NOT: call {{.*}}@llvm.sincos
; CHECK: call float @llvm.roundeven.f32
; CHECK-LABEL: @fast(
; CHECK: call afn float @llvm.exp.f32
; CHECK: call afn float @llvm.log.f32
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

define agx_kernel void @fast(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %a = call afn float @llvm.exp.f32(float %x)
  %b = call afn float @llvm.log.f32(float %a)
  store float %b, ptr addrspace(1) %p
  ret void
}

define agx_kernel void @sc(ptr addrspace(1) %p) {
  %x = load float, ptr addrspace(1) %p
  %sc = call { float, float } @llvm.sincos.f32(float %x)
  %s = extractvalue { float, float } %sc, 0
  %c = extractvalue { float, float } %sc, 1
  %t = fdiv float %s, %c
  store float %t, ptr addrspace(1) %p
  ret void
}

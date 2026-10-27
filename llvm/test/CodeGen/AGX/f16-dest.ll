; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s

; A single precision operation whose only use converts it to half precision
; writes a 16-bit register itself: no separate fadd -0.0 conversion.
; CHECK-LABEL: mul_to_half:
; CHECK:         fmul32 r{{[0-9]+}}l, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK-NOT:     , -0.0
; CHECK:         stop
define agx_kernel void @mul_to_half(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load float, ptr addrspace(1) %p
  %pb = getelementptr float, ptr addrspace(1) %p, i32 1
  %b = load float, ptr addrspace(1) %pb
  %m = fmul float %a, %b
  %h = fptrunc float %m to half
  store half %h, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: fma_to_half:
; CHECK:         fmadd32 r{{[0-9]+}}l, r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
; CHECK-NOT:     , -0.0
; CHECK:         stop
define agx_kernel void @fma_to_half(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load float, ptr addrspace(1) %p
  %pb = getelementptr float, ptr addrspace(1) %p, i32 1
  %b = load float, ptr addrspace(1) %pb
  %pc = getelementptr float, ptr addrspace(1) %p, i32 2
  %c = load float, ptr addrspace(1) %pc
  %m = call float @llvm.fma.f32(float %a, float %b, float %c)
  %h = fptrunc float %m to half
  store half %h, ptr addrspace(1) %q
  ret void
}

; A result with another use keeps its conversion.
; CHECK-LABEL: two_uses:
; CHECK:         fadd32 r{{[0-9]+}}l, r{{[0-9]+}}, -0.0
define agx_kernel void @two_uses(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load float, ptr addrspace(1) %p
  %pb = getelementptr float, ptr addrspace(1) %p, i32 1
  %b = load float, ptr addrspace(1) %pb
  %s = fadd float %a, %b
  %h = fptrunc float %s to half
  store half %h, ptr addrspace(1) %q
  store float %s, ptr addrspace(1) %p
  ret void
}

; Half precision sources stay folded into the operation (FMADD32_M forms).
; CHECK-LABEL: half_sources:
; CHECK:         fmadd32 r{{[0-9]+}}, r{{[0-9]+}}l, r{{[0-9]+}}, r{{[0-9]+}}h
; CHECK:         fadd32 r{{[0-9]+}}l, r{{[0-9]+}}, -0.0
define agx_kernel void @half_sources(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load half, ptr addrspace(1) %q
  %pc = getelementptr half, ptr addrspace(1) %q, i32 1
  %c = load half, ptr addrspace(1) %pc
  %b = load float, ptr addrspace(1) %p
  %ea = fpext half %a to float
  %ec = fpext half %c to float
  %m = call float @llvm.fma.f32(float %ea, float %b, float %ec)
  %h = fptrunc float %m to half
  store half %h, ptr addrspace(1) %q
  ret void
}

declare float @llvm.fma.f32(float, float, float)

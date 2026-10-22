; RUN: llc -mtriple=agx -mcpu=agx-g14c -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g13g -mattr=+agx-strict -verify-machineinstrs < %s | FileCheck %s

; Single precision operations read half precision registers and convert them
; exactly, so extensions from half precision fold into their sources, also
; where an extension has several uses.

; CHECK-LABEL: fma_hh:
; CHECK-NOT:     fadd32
; CHECK:         fmadd32 r{{[0-9]+}}, r{{[0-9]+}}{{[lh]}}, r{{[0-9]+}}{{[lh]}}, r{{[0-9]+}}
; CHECK:         fmadd32 r{{[0-9]+}}, r{{[0-9]+}}{{[lh]}}, r{{[0-9]+}}{{[lh]}}, r{{[0-9]+}}
; CHECK-NOT:     fadd32
define agx_kernel void @fma_hh(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load half, ptr addrspace(1) %p
  %pb = getelementptr half, ptr addrspace(1) %p, i64 1
  %b = load half, ptr addrspace(1) %pb
  %pc = getelementptr half, ptr addrspace(1) %p, i64 2
  %c = load half, ptr addrspace(1) %pc
  %x = load float, ptr addrspace(1) %q
  %ea = fpext half %a to float
  %eb = fpext half %b to float
  %ec = fpext half %c to float
  %f0 = call float @llvm.fma.f32(float %ea, float %eb, float %x)
  %f1 = call float @llvm.fma.f32(float %ea, float %ec, float %f0)
  store float %f1, ptr addrspace(1) %q
  ret void
}

; CHECK-LABEL: mul_neg:
; CHECK:         fmul32 r{{[0-9]+}}, r{{[0-9]+}}{{[lh]}}.neg, r{{[0-9]+$}}
define agx_kernel void @mul_neg(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %a = load half, ptr addrspace(1) %p
  %x = load float, ptr addrspace(1) %q
  %n = fneg half %a
  %ea = fpext half %n to float
  %m = fmul float %ea, %x
  store float %m, ptr addrspace(1) %q
  ret void
}

declare float @llvm.fma.f32(float, float, float)

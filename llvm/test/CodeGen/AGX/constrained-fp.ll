; RUN: llc -mtriple=agx -mcpu=agx-g13g -verify-machineinstrs < %s | FileCheck %s

; There is no floating point environment: constrained operations are
; generated like ordinary ones.

; CHECK-LABEL: strict:
; CHECK: rcp
; CHECK: fmul32
; CHECK: fadd32
; CHECK: fcmpsel lt
; CHECK: device_store
define agx_kernel void @strict(ptr addrspace(1) %out, float %a, float %b) strictfp {
  %d = call float @llvm.experimental.constrained.fdiv.f32(float %a, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %s = call float @llvm.experimental.constrained.fadd.f32(float %d, float %b, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %r = call float @llvm.experimental.constrained.sqrt.f32(float %s, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  %c = call i1 @llvm.experimental.constrained.fcmp.f32(float %r, float %a, metadata !"olt", metadata !"fpexcept.ignore") strictfp
  %v = select i1 %c, float %r, float %a
  store float %v, ptr addrspace(1) %out
  ret void
}

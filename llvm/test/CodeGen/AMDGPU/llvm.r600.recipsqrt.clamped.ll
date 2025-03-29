; RUN: llc -mtriple=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG %s

declare float @llvm.r600.recipsqrt.clamped.f32(float) nounwind readnone

; EG-LABEL: {{^}}rsq_clamped_f32:
; EG: RECIPSQRT_CLAMPED
define amdgpu_kernel void @rsq_clamped_f32(ptr addrspace(1) %out, float %src) nounwind {
  %rsq_clamped = call float @llvm.r600.recipsqrt.clamped.f32(float %src)
  store float %rsq_clamped, ptr addrspace(1) %out, align 4
  ret void
}

; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: Uses of this should be moved to llvm.amdgcn.rsq.clamped, and
; an r600 variant added.

declare float @llvm.AMDGPU.rsq.clamped.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}rsq_clamped_f32:
; SI: v_rsq_clamp_f32_e32

; VI-DAG: v_rsq_f32_e32 [[RSQ:v[0-9]+]], {{s[0-9]+}}
; VI-DAG: v_min_f32_e32 [[MIN:v[0-9]+]], 0x7f7fffff, [[RSQ]]
; TODO: this constant should be folded:
; VI-DAG: v_mov_b32_e32 [[MINFLT:v[0-9]+]], 0xff7fffff
; VI: v_max_f32_e32 {{v[0-9]+}}, [[MIN]], [[MINFLT]]

; EG: RECIPSQRT_CLAMPED

define void @rsq_clamped_f32(float addrspace(1)* %out, float %src) nounwind {
  %rsq_clamped = call float @llvm.AMDGPU.rsq.clamped.f32(float %src) nounwind readnone
  store float %rsq_clamped, float addrspace(1)* %out, align 4
  ret void
}

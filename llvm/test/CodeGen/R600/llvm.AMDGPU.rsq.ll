; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.AMDGPU.rsq.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}rsq_f32:
; SI: v_rsq_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
; EG: RECIPSQRT_IEEE
define void @rsq_f32(float addrspace(1)* %out, float %src) nounwind {
  %rsq = call float @llvm.AMDGPU.rsq.f32(float %src) nounwind readnone
  store float %rsq, float addrspace(1)* %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; FUNC-LABEL: {{^}}rsq_f32_constant_4.0
; SI: v_rsq_f32_e32 {{v[0-9]+}}, 4.0
; EG: RECIPSQRT_IEEE
define void @rsq_f32_constant_4.0(float addrspace(1)* %out) nounwind {
  %rsq = call float @llvm.AMDGPU.rsq.f32(float 4.0) nounwind readnone
  store float %rsq, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_f32_constant_100.0
; SI: v_rsq_f32_e32 {{v[0-9]+}}, 0x42c80000
; EG: RECIPSQRT_IEEE
define void @rsq_f32_constant_100.0(float addrspace(1)* %out) nounwind {
  %rsq = call float @llvm.AMDGPU.rsq.f32(float 100.0) nounwind readnone
  store float %rsq, float addrspace(1)* %out, align 4
  ret void
}

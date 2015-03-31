; RUN: llc -march=amdgcn -mcpu=SI -mattr=-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mcpu=SI -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE-SPDENORM -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE-SPDENORM -check-prefix=SI -check-prefix=FUNC %s

; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG-SAFE -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare float @llvm.AMDGPU.rcp.f32(float) nounwind readnone
declare double @llvm.AMDGPU.rcp.f64(double) nounwind readnone

declare float @llvm.sqrt.f32(float) nounwind readnone

; FUNC-LABEL: {{^}}rcp_f32:
; SI: v_rcp_f32_e32
; EG: RECIP_IEEE
define void @rcp_f32(float addrspace(1)* %out, float %src) nounwind {
  %rcp = call float @llvm.AMDGPU.rcp.f32(float %src) nounwind readnone
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FIXME: Evergreen only ever does unsafe fp math.
; FUNC-LABEL: {{^}}rcp_pat_f32:

; SI-SAFE: v_rcp_f32_e32
; XSI-SAFE-SPDENORM-NOT: v_rcp_f32_e32

; EG: RECIP_IEEE

define void @rcp_pat_f32(float addrspace(1)* %out, float %src) nounwind {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_rcp_pat_f32:
; SI-UNSAFE: v_rsq_f32_e32
; SI-SAFE: v_sqrt_f32_e32
; SI-SAFE: v_rcp_f32_e32

; EG: RECIPSQRT_IEEE
define void @rsq_rcp_pat_f32(float addrspace(1)* %out, float %src) nounwind {
  %sqrt = call float @llvm.sqrt.f32(float %src) nounwind readnone
  %rcp = call float @llvm.AMDGPU.rcp.f32(float %sqrt) nounwind readnone
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

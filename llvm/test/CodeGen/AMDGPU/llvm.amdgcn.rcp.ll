; RUN: llc -march=amdgcn -mattr=-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE-SPDENORM -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals -enable-unsafe-fp-math -verify-machineinstrs < %s | FileCheck -check-prefix=SI-UNSAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; XUN: llc -march=amdgcn -mcpu=tonga -mattr=+fp32-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE-SPDENORM -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.amdgcn.rcp.f32(float) #0
declare double @llvm.amdgcn.rcp.f64(double) #0

declare double @llvm.sqrt.f64(double) #0
declare float @llvm.sqrt.f32(float) #0


; FUNC-LABEL: {{^}}rcp_f32:
; SI: v_rcp_f32_e32
define void @rcp_f32(float addrspace(1)* %out, float %src) #1 {
  %rcp = call float @llvm.amdgcn.rcp.f32(float %src) #0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_pat_f32:

; SI-SAFE: v_rcp_f32_e32
; XSI-SAFE-SPDENORM-NOT: v_rcp_f32_e32
define void @rcp_pat_f32(float addrspace(1)* %out, float %src) #1 {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_rcp_pat_f32:
; SI-UNSAFE: v_rsq_f32_e32
; SI-SAFE: v_sqrt_f32_e32
; SI-SAFE: v_rcp_f32_e32
define void @rsq_rcp_pat_f32(float addrspace(1)* %out, float %src) #1 {
  %sqrt = call float @llvm.sqrt.f32(float %src) #0
  %rcp = call float @llvm.amdgcn.rcp.f32(float %sqrt) #0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_f64:
; SI: v_rcp_f64_e32
define void @rcp_f64(double addrspace(1)* %out, double %src) #1 {
  %rcp = call double @llvm.amdgcn.rcp.f64(double %src) #0
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rcp_pat_f64:
; SI: v_rcp_f64_e32
define void @rcp_pat_f64(double addrspace(1)* %out, double %src) #1 {
  %rcp = fdiv double 1.0, %src
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rsq_rcp_pat_f64:
; SI-UNSAFE: v_rsq_f64_e32
; SI-SAFE-NOT: v_rsq_f64_e32
; SI-SAFE: v_sqrt_f64
; SI-SAFE: v_rcp_f64
define void @rsq_rcp_pat_f64(double addrspace(1)* %out, double %src) #1 {
  %sqrt = call double @llvm.sqrt.f64(double %src) #0
  %rcp = call double @llvm.amdgcn.rcp.f64(double %sqrt) #0
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rcp_undef_f32:
; SI-NOT: v_rcp_f32
define void @rcp_undef_f32(float addrspace(1)* %out) #1 {
  %rcp = call float @llvm.amdgcn.rcp.f32(float undef) #0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

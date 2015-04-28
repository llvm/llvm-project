; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.AMDGPU.rcp.f64(double) nounwind readnone
declare double @llvm.sqrt.f64(double) nounwind readnone

; FUNC-LABEL: {{^}}rcp_f64:
; SI: v_rcp_f64_e32
define void @rcp_f64(double addrspace(1)* %out, double %src) nounwind {
  %rcp = call double @llvm.AMDGPU.rcp.f64(double %src) nounwind readnone
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rcp_pat_f64:
; SI: v_rcp_f64_e32
define void @rcp_pat_f64(double addrspace(1)* %out, double %src) nounwind {
  %rcp = fdiv double 1.0, %src
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}rsq_rcp_pat_f64:
; SI-UNSAFE: v_rsq_f64_e32
; SI-SAFE-NOT: v_rsq_f64_e32
; SI-SAFE: v_sqrt_f64
; SI-SAFE: v_rcp_f64
define void @rsq_rcp_pat_f64(double addrspace(1)* %out, double %src) nounwind {
  %sqrt = call double @llvm.sqrt.f64(double %src) nounwind readnone
  %rcp = call double @llvm.AMDGPU.rcp.f64(double %sqrt) nounwind readnone
  store double %rcp, double addrspace(1)* %out, align 8
  ret void
}

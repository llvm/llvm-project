; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.amdgcn.rsq.legacy(float) #0

; FUNC-LABEL: {{^}}rsq_legacy_f32:
; SI: v_rsq_legacy_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @rsq_legacy_f32(ptr addrspace(1) %out, float %src) #1 {
  %rsq = call float @llvm.amdgcn.rsq.legacy(float %src) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; FUNC-LABEL: {{^}}rsq_legacy_f32_constant_4.0
; SI: v_rsq_legacy_f32_e32 {{v[0-9]+}}, 4.0
define amdgpu_kernel void @rsq_legacy_f32_constant_4.0(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.legacy(float 4.0) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_legacy_f32_constant_100.0
; SI: v_rsq_legacy_f32_e32 {{v[0-9]+}}, 0x42c80000
define amdgpu_kernel void @rsq_legacy_f32_constant_100.0(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.legacy(float 100.0) #0
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rsq_legacy_undef_f32:
; SI-NOT: v_rsq_legacy_f32
define amdgpu_kernel void @rsq_legacy_undef_f32(ptr addrspace(1) %out) #1 {
  %rsq = call float @llvm.amdgcn.rsq.legacy(float undef)
  store float %rsq, ptr addrspace(1) %out, align 4
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

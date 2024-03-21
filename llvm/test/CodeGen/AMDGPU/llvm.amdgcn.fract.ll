; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.fract.f32(float) nounwind readnone
declare double @llvm.amdgcn.fract.f64(double) nounwind readnone

; GCN-LABEL: {{^}}v_fract_f32:
; GCN: v_fract_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @v_fract_f32(ptr addrspace(1) %out, float %src) nounwind {
  %fract = call float @llvm.amdgcn.fract.f32(float %src)
  store float %fract, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_fract_f64:
; GCN: v_fract_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @v_fract_f64(ptr addrspace(1) %out, double %src) nounwind {
  %fract = call double @llvm.amdgcn.fract.f64(double %src)
  store double %fract, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_fract_undef_f32:
; GCN-NOT: v_fract_f32
; GCN-NOT: store_dword
define amdgpu_kernel void @v_fract_undef_f32(ptr addrspace(1) %out) nounwind {
  %fract = call float @llvm.amdgcn.fract.f32(float undef)
  store float %fract, ptr addrspace(1) %out
  ret void
}

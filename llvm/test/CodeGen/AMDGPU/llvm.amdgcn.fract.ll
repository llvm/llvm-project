; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.fract.f32(float) #0
declare double @llvm.amdgcn.fract.f64(double) #0

; GCN-LABEL: {{^}}v_fract_f32:
; GCN: v_fract_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @v_fract_f32(ptr addrspace(1) %out, float %src) #1 {
  %fract = call float @llvm.amdgcn.fract.f32(float %src)
  store float %fract, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_fract_f64:
; GCN: v_fract_f64_e32 {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @v_fract_f64(ptr addrspace(1) %out, double %src) #1 {
  %fract = call double @llvm.amdgcn.fract.f64(double %src)
  store double %fract, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_fract_undef_f32:
; GCN-NOT: v_fract_f32
; GCN-NOT: store_dword
define amdgpu_kernel void @v_fract_undef_f32(ptr addrspace(1) %out) #1 {
  %fract = call float @llvm.amdgcn.fract.f32(float undef)
  store float %fract, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

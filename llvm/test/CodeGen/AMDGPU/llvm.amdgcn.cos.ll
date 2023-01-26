; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare float @llvm.amdgcn.cos.f32(float) #0

; GCN-LABEL: {{^}}v_cos_f32:
; GCN: v_cos_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @v_cos_f32(ptr addrspace(1) %out, float %src) #1 {
  %cos = call float @llvm.amdgcn.cos.f32(float %src) #0
  store float %cos, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

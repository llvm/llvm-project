; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: intrinsic not supported on subtarget

declare float @llvm.amdgcn.log.clamp.f32(float) #0

; GCN-LABEL: {{^}}v_log_clamp_f32:
; GCN: v_log_clamp_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @v_log_clamp_f32(float addrspace(1)* %out, float %src) #1 {
  %log.clamp = call float @llvm.amdgcn.log.clamp.f32(float %src) #0
  store float %log.clamp, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

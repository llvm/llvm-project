; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s

declare float @llvm.amdgcn.sin.f32(float) #0

; GCN-LABEL: {{^}}v_sin_f32:
; GCN: v_sin_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}
define void @v_sin_f32(float addrspace(1)* %out, float %src) #1 {
  %sin = call float @llvm.amdgcn.sin.f32(float %src) #0
  store float %sin, float addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

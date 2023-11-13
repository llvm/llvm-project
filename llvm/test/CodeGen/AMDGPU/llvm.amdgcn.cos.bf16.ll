; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG %s
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1210 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i16 @llvm.amdgcn.cos.bf16(i16) #0

; GCN-LABEL: {{^}}cos_bf16:
; GCN: v_cos_bf16_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @cos_bf16(ptr addrspace(1) %out, i16 %src) #1 {
  %cos = call i16 @llvm.amdgcn.cos.bf16(i16 %src) #0
  store i16 %cos, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}cos_bf16_constant_4
; GCN: v_cos_bf16_e32 {{v[0-9]+}}, 4
define amdgpu_kernel void @cos_bf16_constant_4(ptr addrspace(1) %out) #1 {
  %cos = call i16 @llvm.amdgcn.cos.bf16(i16 4) #0
  store i16 %cos, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}cos_bf16_constant_100
; GCN: v_cos_bf16_e32 {{v[0-9]+}}, 0x64
define amdgpu_kernel void @cos_bf16_constant_100(ptr addrspace(1) %out) #1 {
  %cos = call i16 @llvm.amdgcn.cos.bf16(i16 100) #0
  store i16 %cos, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}cos_undef_bf16:
; SDAG-NOT: v_cos_bf16
define amdgpu_kernel void @cos_undef_bf16(ptr addrspace(1) %out) #1 {
  %cos = call i16 @llvm.amdgcn.cos.bf16(i16 undef)
  store i16 %cos, ptr addrspace(1) %out, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

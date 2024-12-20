; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SDAG %s
; xUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: GlobalISel does not work with bf16

declare bfloat @llvm.amdgcn.rcp.bf16(bfloat) #0

; GCN-LABEL: {{^}}rcp_bf16:
; GCN: v_rcp_bf16_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @rcp_bf16(ptr addrspace(1) %out, bfloat %src) #1 {
  %rcp = call bfloat @llvm.amdgcn.rcp.bf16(bfloat %src) #0
  store bfloat %rcp, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}rcp_bf16_constant_4
; GCN: mov_b32 v{{[0-9]+}}, 0x3e80
define amdgpu_kernel void @rcp_bf16_constant_4(ptr addrspace(1) %out) #1 {
  %rcp = call bfloat @llvm.amdgcn.rcp.bf16(bfloat 4.0) #0
  store bfloat %rcp, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}rcp_bf16_constant_100
; GCN: mov_b32 v{{[0-9]+}}, 0x3c24
define amdgpu_kernel void @rcp_bf16_constant_100(ptr addrspace(1) %out) #1 {
  %rcp = call bfloat @llvm.amdgcn.rcp.bf16(bfloat 100.0) #0
  store bfloat %rcp, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}rcp_undef_bf16:
; SDAG-NOT: v_rcp_bf16
define amdgpu_kernel void @rcp_undef_bf16(ptr addrspace(1) %out) #1 {
  %rcp = call bfloat @llvm.amdgcn.rcp.bf16(bfloat undef)
  store bfloat %rcp, ptr addrspace(1) %out, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

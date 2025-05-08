; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN %s
; xUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: GlobalISel does not work with bf16

declare bfloat @llvm.amdgcn.log.bf16(bfloat) #0

; GCN-LABEL: {{^}}log_bf16:
; GCN: v_log_bf16_e32 {{v[0-9]+}}, {{s[0-9]+}}
define amdgpu_kernel void @log_bf16(ptr addrspace(1) %out, bfloat %src) #1 {
  %log = call bfloat @llvm.amdgcn.log.bf16(bfloat %src) #0
  store bfloat %log, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}log_bf16_constant_4
; GCN: v_log_bf16_e32 v0, 4.0
define amdgpu_kernel void @log_bf16_constant_4(ptr addrspace(1) %out) #1 {
  %log = call bfloat @llvm.amdgcn.log.bf16(bfloat 4.0) #0
  store bfloat %log, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}log_bf16_constant_100
; GCN: v_log_bf16_e32 {{v[0-9]+}}, 0x42c8
define amdgpu_kernel void @log_bf16_constant_100(ptr addrspace(1) %out) #1 {
  %log = call bfloat @llvm.amdgcn.log.bf16(bfloat 100.0) #0
  store bfloat %log, ptr addrspace(1) %out, align 2
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

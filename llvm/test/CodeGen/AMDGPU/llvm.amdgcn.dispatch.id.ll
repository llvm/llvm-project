; RUN: llc -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.dispatch.id() #1

; GCN-LABEL: {{^}}dispatch_id:
; GCN: .amd_kernel_code_t
; GCN: enable_sgpr_dispatch_id = 1

; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s6
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s7
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
define amdgpu_kernel void @dispatch_id(ptr addrspace(1) %out) #0 {
  %tmp0 = call i64 @llvm.amdgcn.dispatch.id()
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

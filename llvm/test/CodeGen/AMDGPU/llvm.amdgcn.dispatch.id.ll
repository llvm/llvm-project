; RUN: llc -global-isel=0 -mtriple=amdgcn--amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -new-reg-bank-select -mtriple=amdgcn--amdhsa < %s | FileCheck -check-prefix=GCN %s

declare i64 @llvm.amdgcn.dispatch.id() #1

; GCN-LABEL: {{^}}dispatch_id:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s10
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s11
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
; GCN: .amdhsa_user_sgpr_dispatch_id 1
define amdgpu_kernel void @dispatch_id(ptr addrspace(1) %out) #0 {
  %tmp0 = call i64 @llvm.amdgcn.dispatch.id()
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dispatch_id_opt0:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s8
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s9
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
; GCN: .amdhsa_user_sgpr_dispatch_id 1
define amdgpu_kernel void @dispatch_id_opt0(ptr addrspace(1) %out) #2 {
  %tmp0 = call i64 @llvm.amdgcn.dispatch.id()
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dispatch_id_opt1:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s6
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s7
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
; GCN: .amdhsa_user_sgpr_dispatch_id 1
define amdgpu_kernel void @dispatch_id_opt1(ptr addrspace(1) %out) #3 {
  %tmp0 = call i64 @llvm.amdgcn.dispatch.id()
  store i64 %tmp0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}dispatch_id_opt2:
; GCN-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s4
; GCN-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s5
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v[[[LO]]:[[HI]]]
; GCN: .amdhsa_user_sgpr_dispatch_id 1
define amdgpu_kernel void @dispatch_id_opt2() #4 {
  %tmp0 = call i64 @llvm.amdgcn.dispatch.id()
  store i64 %tmp0, ptr addrspace(1) null
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { "amdgpu-no-dispatch-ptr" }
attributes #3 = { "amdgpu-no-dispatch-ptr" "amdgpu-no-queue-ptr" }
attributes #4 = { "amdgpu-no-dispatch-ptr" "amdgpu-no-queue-ptr" "amdgpu-no-implicitarg-ptr" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}

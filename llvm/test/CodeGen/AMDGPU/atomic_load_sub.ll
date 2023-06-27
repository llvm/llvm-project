; RUN: llc -march=amdgcn -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -amdgpu-atomic-optimizer-strategy=None < %s | FileCheck -enable-var-scope -check-prefixes=R600,FUNC %s

; FUNC-LABEL: {{^}}atomic_sub_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_SUB *
; GCN: ds_sub_u32
define amdgpu_kernel void @atomic_sub_local(ptr addrspace(3) %local) {
   %unused = atomicrmw volatile sub ptr addrspace(3) %local, i32 5 seq_cst
   ret void
}

; FUNC-LABEL: {{^}}atomic_sub_local_const_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_SUB *
; GCN: ds_sub_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
define amdgpu_kernel void @atomic_sub_local_const_offset(ptr addrspace(3) %local) {
  %gep = getelementptr i32, ptr addrspace(3) %local, i32 4
  %val = atomicrmw volatile sub ptr addrspace(3) %gep, i32 5 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_ret_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_SUB_RET *
; GCN: ds_sub_rtn_u32
define amdgpu_kernel void @atomic_sub_ret_local(ptr addrspace(1) %out, ptr addrspace(3) %local) {
  %val = atomicrmw volatile sub ptr addrspace(3) %local, i32 5 seq_cst
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_ret_local_const_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_SUB_RET *
; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:20
define amdgpu_kernel void @atomic_sub_ret_local_const_offset(ptr addrspace(1) %out, ptr addrspace(3) %local) {
  %gep = getelementptr i32, ptr addrspace(3) %local, i32 5
  %val = atomicrmw volatile sub ptr addrspace(3) %gep, i32 5 seq_cst
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; RUN: llc -march=amdgcn -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -amdgpu-atomic-optimizations=false < %s | FileCheck -check-prefixes=R600,FUNC %s

; FUNC-LABEL: {{^}}atomic_add_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0
; R600: LDS_ADD *
; GCN: ds_add_u32
define amdgpu_kernel void @atomic_add_local(i32 addrspace(3)* %local) {
   %unused = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
   ret void
}

; FUNC-LABEL: {{^}}atomic_add_local_const_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_ADD *
; GCN: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
define amdgpu_kernel void @atomic_add_local_const_offset(i32 addrspace(3)* %local) {
  %gep = getelementptr i32, i32 addrspace(3)* %local, i32 4
  %val = atomicrmw volatile add i32 addrspace(3)* %gep, i32 5 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_ret_local:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_ADD_RET *
; GCN: ds_add_rtn_u32
define amdgpu_kernel void @atomic_add_ret_local(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %val = atomicrmw volatile add i32 addrspace(3)* %local, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_ret_local_const_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; R600: LDS_ADD_RET *
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:20
define amdgpu_kernel void @atomic_add_ret_local_const_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %local) {
  %gep = getelementptr i32, i32 addrspace(3)* %local, i32 5
  %val = atomicrmw volatile add i32 addrspace(3)* %gep, i32 5 seq_cst
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

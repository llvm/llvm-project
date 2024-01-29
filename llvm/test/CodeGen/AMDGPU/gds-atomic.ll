; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,FUNC %s

; FUNC-LABEL: {{^}}atomic_add_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_add_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw volatile add ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_add_ret_gds_const_offset:
; GCN: s_movk_i32 m0, 0x80
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:20 gds
define amdgpu_kernel void @atomic_add_ret_gds_const_offset(ptr addrspace(1) %out, ptr addrspace(2) %gds) #0 {
  %gep = getelementptr i32, ptr addrspace(2) %gds, i32 5
  %val = atomicrmw volatile add ptr addrspace(2) %gep, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_sub_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_sub_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw sub ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_and_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_and_rtn_b32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_and_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw and ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_or_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_or_rtn_b32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_or_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw or ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_xor_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_xor_rtn_b32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_xor_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw xor ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_umin_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_min_rtn_u32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_umin_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw umin ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_umax_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_max_rtn_u32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_umax_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw umax ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_imin_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_min_rtn_i32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_imin_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw min ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_imax_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_max_rtn_i32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_imax_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw max ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_xchg_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_wrxchg_rtn_b32 v{{[0-9]+}}, v[[OFF]], v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_xchg_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw xchg ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_ret_gds:
; GCN-DAG: v_mov_b32_e32 v[[OFF:[0-9]+]], s
; GCN-DAG: s_movk_i32 m0, 0x1000
; GCN: ds_cmpst_rtn_b32 v{{[0-9]+}}, v[[OFF:[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}} gds
define amdgpu_kernel void @atomic_cmpxchg_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = cmpxchg ptr addrspace(2) %gds, i32 0, i32 1 acquire acquire
  %x = extractvalue { i32, i1 } %val, 0
  store i32 %x, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind "amdgpu-gds-size"="128" }
attributes #1 = { nounwind "amdgpu-gds-size"="4096" }

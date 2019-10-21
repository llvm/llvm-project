; RUN: llc -march=amdgcn -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI,SIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -amdgpu-atomic-optimizations=false -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,SIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}atomic_add_i32_offset:
; SIVI: buffer_atomic_add v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}
; GFX9: global_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_add_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_max_neg_offset:
; GFX9: global_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:-4096{{$}}
define amdgpu_kernel void @atomic_add_i32_max_neg_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 -1024
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_soffset:
; SIVI: s_mov_b32 [[SREG:s[0-9]+]], 0x8ca0
; SIVI: buffer_atomic_add v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], [[SREG]]{{$}}

; GFX9: v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0x8000,
; GFX9-NEXT: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
; GFX9: global_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:3232{{$}}
define amdgpu_kernel void @atomic_add_i32_soffset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 9000
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_huge_offset:
; SI-DAG: v_mov_b32_e32 v[[PTRLO:[0-9]+]], 0xdeac
; SI-DAG: v_mov_b32_e32 v[[PTRHI:[0-9]+]], 0xabcd
; SI: buffer_atomic_add v{{[0-9]+}}, v{{\[}}[[PTRLO]]:[[PTRHI]]{{\]}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}

; VI: flat_atomic_add

; GFX9: v_mov_b32_e32 [[HIGH_K:v[0-9]+]], 0xabcd
; GFX9: v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0xd000,
; GFX9-NEXT: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, [[HIGH_K]], v{{[0-9]+}}, vcc
; GFX9: global_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:3756{{$}}
define amdgpu_kernel void @atomic_add_i32_huge_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 47224239175595

  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_offset:
; SIVI: buffer_atomic_add [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_add v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_add_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64_offset:
; SI: buffer_atomic_add v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_add_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64_offset:
; SI: buffer_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
; GFX9: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile add i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32:
; SIVI: buffer_atomic_add v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_add_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile add i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret:
; SIVI: buffer_atomic_add [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
; GFX9: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile add i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64:
; SI: buffer_atomic_add v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_add_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64:
; SI: buffer_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_add_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_offset:
; SIVI: buffer_atomic_and v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_and_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_offset:
; SIVI: buffer_atomic_and [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_and_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64_offset:
; SI: buffer_atomic_and v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_and_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64_offset:
; SI: buffer_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_and_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile and i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32:
; SIVI: buffer_atomic_and v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_and v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_and_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile and i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret:
; SIVI: buffer_atomic_and [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_and v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_and_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile and i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64:
; SI: buffer_atomic_and v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_and_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64:
; SI: buffer_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_and_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_offset:
; SIVI: buffer_atomic_sub v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_sub_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_offset:
; SIVI: buffer_atomic_sub [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_sub v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_sub_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64_offset:
; SI: buffer_atomic_sub v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_sub_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64_offset:
; SI: buffer_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_sub_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile sub i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32:
; SIVI: buffer_atomic_sub v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_sub v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_sub_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret:
; SIVI: buffer_atomic_sub [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_sub_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64:
; SI: buffer_atomic_sub v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_sub v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_sub_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64:
; SI: buffer_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_sub_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_offset:
; SIVI: buffer_atomic_smax v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_max_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_offset:
; SIVI: buffer_atomic_smax [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_max_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64_offset:
; SI: buffer_atomic_smax v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_max_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64_offset:
; SI: buffer_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_max_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile max i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32:
; SIVI: buffer_atomic_smax v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_max_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile max i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret:
; SIVI: buffer_atomic_smax [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_max_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile max i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64:
; SI: buffer_atomic_smax v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}

; GFX9: global_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_max_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64:
; SI: buffer_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_max_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_offset:
; SIVI: buffer_atomic_umax v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_umax_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_offset:
; SIVI: buffer_atomic_umax [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_umax_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64_offset:
; SI: buffer_atomic_umax v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_umax_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64_offset:
; SI: buffer_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_umax_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umax i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32:
; SIVI: buffer_atomic_umax v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_umax_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret:
; SIVI: buffer_atomic_umax [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_umax_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64:
; SI: buffer_atomic_umax v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_umax_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64:
; SI: buffer_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_umax_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_offset:
; SIVI: buffer_atomic_smin v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_min_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_offset:
; SIVI: buffer_atomic_smin [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_min_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64_offset:
; SI: buffer_atomic_smin v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16
define amdgpu_kernel void @atomic_min_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64_offset:
; SI: buffer_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_min_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile min i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32:
; SIVI: buffer_atomic_smin v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_min_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile min i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret:
; SIVI: buffer_atomic_smin [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_min_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile min i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64:
; SI: buffer_atomic_smin v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_min_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64:
; SI: buffer_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_min_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile min i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_offset:
; SIVI: buffer_atomic_umin v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_umin_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_offset:
; SIVI: buffer_atomic_umin [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_umin_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64_offset:
; SI: buffer_atomic_umin v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_umin_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64_offset:
; SI: buffer_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_umin_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile umin i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32:
; SIVI: buffer_atomic_umin v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_umin_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret:
; SIVI: buffer_atomic_umin [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_umin_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64:
; SI: buffer_atomic_umin v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_umin_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64:
; SI: buffer_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_umin_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile umin i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_offset:
; SIVI: buffer_atomic_or v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_or_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_offset:
; SIVI: buffer_atomic_or [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_or_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64_offset:
; SI: buffer_atomic_or v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16
define amdgpu_kernel void @atomic_or_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64_offset:
; SI: buffer_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_or_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile or i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32:
; SIVI: buffer_atomic_or v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_or_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile or i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret:
; SIVI: buffer_atomic_or [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_or_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile or i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64:
; SI: buffer_atomic_or v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_or_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64:
; SI: buffer_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_or_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile or i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_offset:
; SIVI: buffer_atomic_swap v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_f32_offset:
; SIVI: buffer_atomic_swap v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_f32_offset(float addrspace(1)* %out, float %in) {
entry:
  %gep = getelementptr float, float addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xchg float addrspace(1)* %gep, float %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_offset:
; SIVI: buffer_atomic_swap [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_xchg_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64_offset:
; SI: buffer_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64_offset:
; SI: buffer_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_xchg_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xchg i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32:
; SIVI: buffer_atomic_swap v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_xchg_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret:
; SIVI: buffer_atomic_swap [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_xchg_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64:
; SI: buffer_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_xchg_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64:
; SI: buffer_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_swap [[RET:v[0-9]+]],  v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_swap [[RET:v[0-9]+]],  v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_xchg_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xchg i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_offset:
; SIVI: buffer_atomic_cmpswap v[{{[0-9]+}}:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:16{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_offset(i32 addrspace(1)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_offset:
; SIVI: buffer_atomic_cmpswap v{{\[}}[[RET:[0-9]+]]{{:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword v[[RET]]

; GFX9: global_atomic_cmpswap [[RET:v[0-9]+]],  v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  %extract0 = extractvalue { i32, i1 } %val, 0
  store i32 %extract0, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64_offset:
; SI: buffer_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}

; VI: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: global_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], off offset:16{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64_offset:
; SI: buffer_atomic_cmpswap v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; SIVI: buffer_store_dword v[[RET]]

; GFX9: global_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i32 addrspace(1)* %gep, i32 %old, i32 %in seq_cst seq_cst
  %extract0 = extractvalue { i32, i1 } %val, 0
  store i32 %extract0, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32:
; SIVI: buffer_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}

; GFX9: global_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32(i32 addrspace(1)* %out, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32 addrspace(1)* %out, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret:
; SIVI: buffer_atomic_cmpswap v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword v[[RET]]

; GFX9: global_atomic_cmpswap [[RET:v[0-9]+]], v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32 addrspace(1)* %out, i32 %old, i32 %in seq_cst seq_cst
  %extract0 = extractvalue { i32, i1 } %val, 0
  store i32 %extract0, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64:
; SI: buffer_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
; GFX9: global_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i32 addrspace(1)* %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64:
; SI: buffer_atomic_cmpswap v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; SIVI: buffer_store_dword v[[RET]]

; GFX9: global_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i32 addrspace(1)* %ptr, i32 %old, i32 %in seq_cst seq_cst
  %extract0 = extractvalue { i32, i1 } %val, 0
  store i32 %extract0, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_offset:
; SIVI: buffer_atomic_xor v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}

; GFX9: global_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_xor_i32_offset(i32 addrspace(1)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_offset:
; SIVI: buffer_atomic_xor [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_xor v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_xor_i32_ret_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64_offset:
; SI: buffer_atomic_xor v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_xor_i32_addr64_offset(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64_offset:
; SI: buffer_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_xor_i32_ret_addr64_offset(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = atomicrmw volatile xor i32 addrspace(1)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32:
; SIVI: buffer_atomic_xor v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_xor_i32(i32 addrspace(1)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32 addrspace(1)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret:
; SIVI: buffer_atomic_xor [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_xor_i32_ret(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32 addrspace(1)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64:
; SI: buffer_atomic_xor v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: global_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_xor_i32_addr64(i32 addrspace(1)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64:
; SI: buffer_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, off glc{{$}}
define amdgpu_kernel void @atomic_xor_i32_ret_addr64(i32 addrspace(1)* %out, i32 addrspace(1)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %val = atomicrmw volatile xor i32 addrspace(1)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_offset:
; SI: buffer_load_dword [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16 glc{{$}}
; VI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_load_i32_offset(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %in, i64 4
  %val = load atomic i32, i32 addrspace(1)* %gep  seq_cst, align 4
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32:
; SI: buffer_load_dword [[RET:v[0-9]+]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; VI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], off glc
define amdgpu_kernel void @atomic_load_i32(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(1)* %in seq_cst, align 4
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64_offset:
; SI: buffer_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16 glc{{$}}
; VI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], off offset:16 glc{{$}}
define amdgpu_kernel void @atomic_load_i32_addr64_offset(i32 addrspace(1)* %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  %val = load atomic i32, i32 addrspace(1)* %gep seq_cst, align 4
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64:
; SI: buffer_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; SIVI: buffer_store_dword [[RET]]

; GFX9: global_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_load_i32_addr64(i32 addrspace(1)* %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %in, i64 %index
  %val = load atomic i32, i32 addrspace(1)* %ptr seq_cst, align 4
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_offset:
; SI: buffer_store_dword {{v[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:16{{$}}
; VI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+$}}
; GFX9: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_store_i32_offset(i32 %in, i32 addrspace(1)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(1)* %out, i64 4
  store atomic i32 %in, i32 addrspace(1)* %gep  seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32:
; SI: buffer_store_dword {{v[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; VI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+$}}
; GFX9: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_store_i32(i32 %in, i32 addrspace(1)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(1)* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64_offset:
; SI: buffer_store_dword {{v[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:16{{$}}
; VI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+$}}
; GFX9: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}, off offset:16{{$}}
define amdgpu_kernel void @atomic_store_i32_addr64_offset(i32 %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(1)* %ptr, i64 4
  store atomic i32 %in, i32 addrspace(1)* %gep seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64:
; SI: buffer_store_dword {{v[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+$}}
; GFX9: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}, off{{$}}
define amdgpu_kernel void @atomic_store_i32_addr64(i32 %in, i32 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(1)* %out, i64 %index
  store atomic i32 %in, i32 addrspace(1)* %ptr seq_cst, align 4
  ret void
}

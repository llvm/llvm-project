; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,CIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}atomic_add_i32_offset:
; CIVI: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX9: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_add_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_max_offset:
; CIVI: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX9: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset:4092{{$}}
define amdgpu_kernel void @atomic_add_i32_max_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 1023
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_max_offset_p1:
; GCN: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_add_i32_max_offset_p1(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 1024
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_offset:
; CIVI: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64_offset:
; CIVI: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_add_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64_offset:
; CIVI: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile add i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32:
; GCN: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_add_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile add i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile add i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64:
; GCN: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_add_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile add i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile add i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_offset:
; CIVI: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_and_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile and i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_offset:
; CIVI: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile and i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64_offset:
; CIVI: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_and_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile and i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64_offset:
; CIVI: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile and i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_and_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile and i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret:
; GCN: flat_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile and i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_and_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile and i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64:
; GCN: flat_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile and i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_offset:
; CIVI: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_sub_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile sub i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_offset:
; CIVI: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile sub i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64_offset:
; CIVI: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_sub_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile sub i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64_offset:
; CIVI: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile sub i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_sub_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_sub_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile sub i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile sub i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_offset:
; CIVI: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_max_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile max i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_offset:
; CIVI: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile max i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64_offset:
; CIVI: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_max_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile max i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64_offset:
; CIVI: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile max i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_max_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile max i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile max i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_max_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile max i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile max i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_offset:
; CIVI: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_umax_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile umax i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_offset:
; CIVI: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile umax i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64_offset:
; CIVI: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_umax_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile umax i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64_offset:
; CIVI: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile umax i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_umax_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_umax_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile umax i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile umax i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_offset:
; CIVI: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_min_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile min i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_offset:
; CIVI: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile min i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64_offset:
; CIVI: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_min_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile min i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64_offset:
; CIVI: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile min i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_min_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile min i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile min i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_min_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile min i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile min i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_offset:
; CIVI: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_umin_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile umin i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_offset:
; CIVI: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile umin i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64_offset:
; CIVI: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_umin_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile umin i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64_offset:
; CIVI: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile umin i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_umin_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret:
; GCN: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_umin_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile umin i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64:
; GCN: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]{{$}}
  define amdgpu_kernel void @atomic_umin_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile umin i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_offset:
; CIVI: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_or_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile or i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_offset:
; CIVI: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile or i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64_offset:
; CIVI: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_or_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile or i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64_offset:
; CIVI: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile or i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_or_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile or i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile or i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_or_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile or i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile or i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_offset:
; CIVI: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile xchg i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_f32_offset:
; CIVI: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_f32_offset(float* %out, float %in) {
entry:
  %gep = getelementptr float, float* %out, i32 4
  %val = atomicrmw volatile xchg float* %gep, float %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_offset:
; CIVI: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile xchg i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64_offset:
; CIVI: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_xchg_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile xchg i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64_offset:
; CIVI: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile xchg i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32:
; GCN: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_xchg_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret:
; GCN: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64:
; GCN: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_xchg_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile xchg i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64:
; GCN: flat_atomic_swap [[RET:v[0-9]+]],  v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile xchg i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; CMP_SWAP

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_offset:
; CIVI: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] offset:16{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_offset(i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_offset:
; CIVI: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX9: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_offset(i32* %out, i32* %out2, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64_offset:
; CIVI: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] offset:16{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64_offset(i32* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val  = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64_offset:
; CIVI: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GFX9: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val  = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32:
; GCN: flat_atomic_cmpswap v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32(i32* %out, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32* %out, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define amdgpu_kernel void @atomic_cmpxchg_i32_ret(i32* %out, i32* %out2, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32* %out, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64:
; GCN: flat_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i32_addr64(i32* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = cmpxchg volatile i32* %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define amdgpu_kernel void @atomic_cmpxchg_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = cmpxchg volatile i32* %ptr, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_offset:
; CIVI: flat_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX9: flat_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_xor_i32_offset(i32* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile xor i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_offset:
; CIVI: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i32_ret_offset(i32* %out, i32* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = atomicrmw volatile xor i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64_offset:
; CIVI: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
; GFX9: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_xor_i32_addr64_offset(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile xor i32* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64_offset:
; CIVI: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX9: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i32_ret_addr64_offset(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = atomicrmw volatile xor i32* %gep, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32:
; GCN: flat_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_xor_i32(i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i32_ret(i32* %out, i32* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64:
; GCN: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define amdgpu_kernel void @atomic_xor_i32_addr64(i32* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile xor i32* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i32_ret_addr64(i32* %out, i32* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %val = atomicrmw volatile xor i32* %ptr, i32 %in seq_cst
  store i32 %val, i32* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_offset:
; CIVI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX9: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i32_offset(i32* %in, i32* %out) {
entry:
  %gep = getelementptr i32, i32* %in, i32 4
  %val = load atomic i32, i32* %gep  seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i32(i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64_offset:
; CIVI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GFX9: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i32_addr64_offset(i32* %in, i32* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %in, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  %val = load atomic i32, i32* %gep seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i32_addr64(i32* %in, i32* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %in, i64 %index
  %val = load atomic i32, i32* %ptr seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_offset:
; CIVI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i32_offset(i32 %in, i32* %out) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  store atomic i32 %in, i32* %gep  seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_i32(i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64_offset:
; CIVI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i32_addr64_offset(i32 %in, i32* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  %gep = getelementptr i32, i32* %ptr, i32 4
  store atomic i32 %in, i32* %gep seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_i32_addr64(i32 %in, i32* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32* %out, i64 %index
  store atomic i32 %in, i32* %ptr seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_load_f32_offset:
; CIVI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX9: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_f32_offset(float* %in, float* %out) {
entry:
  %gep = getelementptr float, float* %in, i32 4
  %val = load atomic float, float* %gep  seq_cst, align 4
  store float %val, float* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_f32:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_f32(float* %in, float* %out) {
entry:
  %val = load atomic float, float* %in seq_cst, align 4
  store float %val, float* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_f32_addr64_offset:
; CIVI: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GFX9: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_f32_addr64_offset(float* %in, float* %out, i64 %index) {
entry:
  %ptr = getelementptr float, float* %in, i64 %index
  %gep = getelementptr float, float* %ptr, i32 4
  %val = load atomic float, float* %gep seq_cst, align 4
  store float %val, float* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_f32_addr64:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_f32_addr64(float* %in, float* %out, i64 %index) {
entry:
  %ptr = getelementptr float, float* %in, i64 %index
  %val = load atomic float, float* %ptr seq_cst, align 4
  store float %val, float* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f32_offset:
; CIVI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_f32_offset(float %in, float* %out) {
entry:
  %gep = getelementptr float, float* %out, i32 4
  store atomic float %in, float* %gep  seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f32:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_f32(float %in, float* %out) {
entry:
  store atomic float %in, float* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f32_addr64_offset:
; CIVI: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_f32_addr64_offset(float %in, float* %out, i64 %index) {
entry:
  %ptr = getelementptr float, float* %out, i64 %index
  %gep = getelementptr float, float* %ptr, i32 4
  store atomic float %in, float* %gep seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f32_addr64:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_f32_addr64(float %in, float* %out, i64 %index) {
entry:
  %ptr = getelementptr float, float* %out, i64 %index
  store atomic float %in, float* %ptr seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i8_offset:
; CIVI: flat_load_ubyte [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX9: flat_load_ubyte [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i8_offset(i8* %in, i8* %out) {
entry:
  %gep = getelementptr i8, i8* %in, i64 16
  %val = load atomic i8, i8* %gep  seq_cst, align 1
  store i8 %val, i8* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i8:
; GCN: flat_load_ubyte [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i8(i8* %in, i8* %out) {
entry:
  %val = load atomic i8, i8* %in seq_cst, align 1
  store i8 %val, i8* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i8_addr64_offset:
; CIVI: flat_load_ubyte [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GFX9: flat_load_ubyte [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_byte v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i8_addr64_offset(i8* %in, i8* %out, i64 %index) {
entry:
  %ptr = getelementptr i8, i8* %in, i64 %index
  %gep = getelementptr i8, i8* %ptr, i64 16
  %val = load atomic i8, i8* %gep seq_cst, align 1
  store i8 %val, i8* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i8_offset:
; CIVI: flat_store_byte v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_byte v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i8_offset(i8 %in, i8* %out) {
entry:
  %gep = getelementptr i8, i8* %out, i64 16
  store atomic i8 %in, i8* %gep  seq_cst, align 1
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i8:
; GCN: flat_store_byte v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_i8(i8 %in, i8* %out) {
entry:
  store atomic i8 %in, i8* %out seq_cst, align 1
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i8_addr64_offset:
; CIVI: flat_store_byte v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_byte v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i8_addr64_offset(i8 %in, i8* %out, i64 %index) {
entry:
  %ptr = getelementptr i8, i8* %out, i64 %index
  %gep = getelementptr i8, i8* %ptr, i64 16
  store atomic i8 %in, i8* %gep seq_cst, align 1
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i16_offset:
; CIVI: flat_load_ushort [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX9: flat_load_ushort [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i16_offset(i16* %in, i16* %out) {
entry:
  %gep = getelementptr i16, i16* %in, i64 8
  %val = load atomic i16, i16* %gep  seq_cst, align 2
  store i16 %val, i16* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i16:
; GCN: flat_load_ushort [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i16(i16* %in, i16* %out) {
entry:
  %val = load atomic i16, i16* %in seq_cst, align 2
  store i16 %val, i16* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i16_addr64_offset:
; CIVI: flat_load_ushort [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GFX9: flat_load_ushort [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] offset:16 glc{{$}}
; GCN: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i16_addr64_offset(i16* %in, i16* %out, i64 %index) {
entry:
  %ptr = getelementptr i16, i16* %in, i64 %index
  %gep = getelementptr i16, i16* %ptr, i64 8
  %val = load atomic i16, i16* %gep seq_cst, align 2
  store i16 %val, i16* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i16_offset:
; CIVI: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i16_offset(i16 %in, i16* %out) {
entry:
  %gep = getelementptr i16, i16* %out, i64 8
  store atomic i16 %in, i16* %gep  seq_cst, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i16:
; GCN: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_i16(i16 %in, i16* %out) {
entry:
  store atomic i16 %in, i16* %out seq_cst, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i16_addr64_offset:
; CIVI: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_i16_addr64_offset(i16 %in, i16* %out, i64 %index) {
entry:
  %ptr = getelementptr i16, i16* %out, i64 %index
  %gep = getelementptr i16, i16* %ptr, i64 8
  store atomic i16 %in, i16* %gep seq_cst, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f16_offset:
; CIVI: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX9: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}} offset:16{{$}}
define amdgpu_kernel void @atomic_store_f16_offset(half %in, half* %out) {
entry:
  %gep = getelementptr half, half* %out, i64 8
  store atomic half %in, half* %gep  seq_cst, align 2
  ret void
}

; GCN-LABEL: {{^}}atomic_store_f16:
; GCN: flat_store_short v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @atomic_store_f16(half %in, half* %out) {
entry:
  store atomic half %in, half* %out seq_cst, align 2
  ret void
}

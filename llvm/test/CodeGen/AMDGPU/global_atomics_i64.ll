; RUN: llc -march=amdgcn -mcpu=bonaire -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -amdgpu-atomic-optimizations=false -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,CIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -amdgpu-atomic-optimizations=false -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; GCN-LABEL: {{^}}atomic_add_i64_offset:
; CIVI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}

; GFX9: global_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], off offset:32{{$}}
define amdgpu_kernel void @atomic_add_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_offset:
; CIVI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_add_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64_offset:
; CI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}{{$}}
; GFX9: global_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_add_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64_offset:
; CI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_add_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64:
; SIVI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_add_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret:
; CIVI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_add_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64:
; CI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_add_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64:
; CI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_add_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_offset:
; CIVI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_and_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_offset:
; CIVI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_and_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64_offset:
; CI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_and_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64_offset:
; CI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_and_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64:
; CIVI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_and_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret:
; CIVI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_and_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64:
; CI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_and_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64:
; CI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_and_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_offset:
; CIVI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_sub_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_offset:
; CIVI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_sub_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64_offset:
; CI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_sub_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64_offset:
; CI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_sub_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64:
; CIVI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_sub_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret:
; CIVI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_sub_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64:
; CI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_sub_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64:
; CI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_sub_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_offset:
; CIVI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_max_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_offset:
; CIVI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_max_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64_offset:
; CI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_max_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64_offset:
; CI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_max_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64:
; CIVI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_max_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret:
; CIVI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_max_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64:
; CI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_max_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64:
; CI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_max_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_offset:
; CIVI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_umax_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_offset:
; CIVI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_umax_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64_offset:
; CI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; FX9: global_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} offset:32{{$}}
define amdgpu_kernel void @atomic_umax_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64_offset:
; CI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_umax_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64:
; CIVI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_umax_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret:
; CIVI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_umax_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64:
; CI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_umax_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64:
; CI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_umax_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_offset:
; CIVI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_min_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_offset:
; CIVI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_min_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64_offset:
; CI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_min_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64_offset:
; CI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_min_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64:
; CIVI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_min_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret:
; CIVI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_min_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64:
; CI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_min_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64:
; CI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_min_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_offset:
; CIVI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}

; GFX9: global_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_umin_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_offset:
; CIVI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_umin_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64_offset:
; CI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_umin_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64_offset:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_umin_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64:
; CIVI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_umin_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret:
; CIVI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_umin_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64:
; CI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_umin_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_umin_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_offset:
; CIVI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_or_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_offset:
; CIVI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_or_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64_offset:
; CI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_or_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64_offset:
; CI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_or_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64:
; CIVI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_or_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret:
; CIVI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_or_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64:
; CI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_or_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64:
; CI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_or_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_offset:
; CIVI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}

; GFX9: global_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_xchg_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_f64_offset:
; CIVI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}

; GFX9: global_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_xchg_f64_offset(double addrspace(1)* %out, double %in) {
entry:
  %gep = getelementptr double, double addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg double addrspace(1)* %gep, double %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_offset:
; CIVI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_xchg_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64_offset:
; CI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}{{$}}
; GFX9: global_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_xchg_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64_offset:
; CI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64:
; CIVI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_xchg_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret:
; CIVI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_xchg_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64:
; CI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_xchg_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64:
; CI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]],  v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_offset:
; CIVI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_xor_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_offset:
; CIVI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_xor_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64_offset:
; CI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_xor_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64_offset:
; CI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_xor_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64:
; CIVI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_xor_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret:
; CIVI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_xor_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64:
; CI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
; GFX9: global_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off{{$}}
define amdgpu_kernel void @atomic_xor_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64:
; CI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off glc{{$}}
define amdgpu_kernel void @atomic_xor_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}


; GCN-LABEL: {{^}}atomic_cmpxchg_i64_offset:
; CIVI: buffer_atomic_cmpswap_x2 v[{{[0-9]+}}:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; GFX9: global_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_offset(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_soffset:
; CIVI: s_mov_b32 [[SREG:s[0-9]+]], 0x11940
; CIVI: buffer_atomic_cmpswap_x2 v[{{[0-9]+}}:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], [[SREG]]{{$}}

; GFX9: v_add_co_u32_e32 v{{[0-9]+}}, vcc, 0x11000,
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
; GFX9: global_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:2368{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_soffset(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 9000
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_offset:
; CIVI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]{{:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; CIVI: buffer_store_dwordx2 v{{\[}}[[RET]]:

; GFX9: global_atomic_cmpswap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}, off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_addr64_offset:
; CI: buffer_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: global_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], off offset:32{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64_offset:
; CI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; CIVI: buffer_store_dwordx2 v{{\[}}[[RET]]:

; GFX9: global_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64:
; CIVI: buffer_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; GFX9: global_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(1)* %out, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret:
; CIVI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; CIVI: buffer_store_dwordx2 v{{\[}}[[RET]]:

; GFX9: global_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(1)* %out, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_addr64:
; CI: buffer_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
; GFX9: global_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(1)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64:
; CI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; CIVI: buffer_store_dwordx2 v{{\[}}[[RET]]:

; GFX9: global_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(1)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_offset:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}], off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_load_i64_offset(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %in, i64 4
  %val = load atomic i64, i64 addrspace(1)* %gep  seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_load_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %val = load atomic i64, i64 addrspace(1)* %in seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_addr64_offset:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], off offset:32 glc{{$}}
define amdgpu_kernel void @atomic_load_i64_addr64_offset(i64 addrspace(1)* %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = load atomic i64, i64 addrspace(1)* %gep seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_addr64:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; CIVI: buffer_store_dwordx2 [[RET]]

; GFX9: global_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], off glc{{$}}
define amdgpu_kernel void @atomic_load_i64_addr64(i64 addrspace(1)* %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %index
  %val = load atomic i64, i64 addrspace(1)* %ptr seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_offset:
; CI: buffer_store_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; VI: flat_store_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: global_store_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}], off offset:32{{$}}
define amdgpu_kernel void @atomic_store_i64_offset(i64 %in, i64 addrspace(1)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  store atomic i64 %in, i64 addrspace(1)* %gep  seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX9: global_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_store_i64(i64 %in, i64 addrspace(1)* %out) {
entry:
  store atomic i64 %in, i64 addrspace(1)* %out seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_addr64_offset:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
; GFX9: global_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], off offset:32{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64_offset(i64 %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  store atomic i64 %in, i64 addrspace(1)* %gep seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_addr64:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
; GFX9: global_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64(i64 %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  store atomic i64 %in, i64 addrspace(1)* %ptr seq_cst, align 8
  ret void
}

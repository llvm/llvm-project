; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}atomic_add_i64_offset:
; GCN: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_add_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_offset:
; GCN: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64:
; GCN: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_add_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret:
; GCN: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64:
; CI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_add_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64:
; CI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_offset:
; GCN: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_and_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_offset:
; GCN: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64:
; GCN: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_and_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret:
; GCN: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64:
; CI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_and_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64:
; CI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_offset:
; GCN: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_sub_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_offset:
; GCN: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64:
; GCN: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_sub_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret:
; GCN: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64:
; CI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_sub_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64:
; CI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_offset:
; GCN: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_max_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_offset:
; GCN: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64:
; GCN: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_max_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret:
; GCN: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64:
; CI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_max_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64:
; CI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_offset:
; GCN: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_umax_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_offset:
; GCN: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64:
; GCN: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_umax_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret:
; GCN: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64:
; CI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umax_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64:
; CI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_offset:
; GCN: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_min_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_offset:
; GCN: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64:
; GCN: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_min_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret:
; GCN: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64:
; CI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_min_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64:
; CI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_offset:
; GCN: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_umin_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_offset:
; GCN: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64:
; GCN: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_umin_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64:
; CI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umin_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_offset:
; GCN: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_or_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_offset:
; GCN: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64:
; GCN: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_or_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret:
; GCN: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64:
; CI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_or_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64:
; CI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_offset:
; GCN: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_xchg_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_offset:
; GCN: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64:
; GCN: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_xchg_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret:
; GCN: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64:
; CI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xchg_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64:
; CI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]],  v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_offset:
; GCN: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_xor_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_offset:
; GCN: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
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
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64:
; GCN: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_xor_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret:
; GCN: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64:
; CI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xor_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64:
; CI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}









; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_offset:
; GCN: buffer_atomic_cmpswap_x2 v[{{[0-9]+}}:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_offset(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_soffset:
; GCN: s_mov_b32 [[SREG:s[0-9]+]], 0x11940
; GCN: buffer_atomic_cmpswap_x2 v[{{[0-9]+}}:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], [[SREG]]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_soffset(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 9000
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_ret_offset:
; GCN: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]{{:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_addr64_offset:
; CI: buffer_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}

; VI: flat_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64_offset:
; CI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(1)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64:
; GCN: buffer_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64(i64 addrspace(1)* %out, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(1)* %out, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_ret:
; GCN: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(1)* %out, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_addr64:
; CI: buffer_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(1)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; FUNC-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64:
; CI: buffer_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(1)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(1)* %out2
  ret void
}

; FUNC-LABEL: {{^}}atomic_load_i64_offset:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_load_i64_offset(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %in, i64 4
  %val = load atomic i64, i64 addrspace(1)* %gep  seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_load_i64:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_load_i64(i64 addrspace(1)* %in, i64 addrspace(1)* %out) {
entry:
  %val = load atomic i64, i64 addrspace(1)* %in seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_load_i64_addr64_offset:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_load_i64_addr64_offset(i64 addrspace(1)* %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %val = load atomic i64, i64 addrspace(1)* %gep seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_load_i64_addr64:
; CI: buffer_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define amdgpu_kernel void @atomic_load_i64_addr64(i64 addrspace(1)* %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %in, i64 %index
  %val = load atomic i64, i64 addrspace(1)* %ptr seq_cst, align 8
  store i64 %val, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i64_offset:
; CI: buffer_store_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
; VI: flat_store_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_offset(i64 %in, i64 addrspace(1)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  store atomic i64 %in, i64 addrspace(1)* %gep  seq_cst, align 8
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i64:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64(i64 %in, i64 addrspace(1)* %out) {
entry:
  store atomic i64 %in, i64 addrspace(1)* %out seq_cst, align 8
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i64_addr64_offset:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64_offset(i64 %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  store atomic i64 %in, i64 addrspace(1)* %gep seq_cst, align 8
  ret void
}

; FUNC-LABEL: {{^}}atomic_store_i64_addr64:
; CI: buffer_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64(i64 %in, i64 addrspace(1)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  store atomic i64 %in, i64 addrspace(1)* %ptr seq_cst, align 8
  ret void
}

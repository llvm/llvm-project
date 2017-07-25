; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}atomic_add_i64_offset:
; GCN: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}{{$}}
define amdgpu_kernel void @atomic_add_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_offset:
; GCN: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64_offset:
; GCN: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}{{$}}
define amdgpu_kernel void @atomic_add_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64_offset:
; GCN: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64:
; GCN: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_add_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret:
; GCN: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64:
; GCN: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_add_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64:
; GCN: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_add_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile add i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_offset:
; GCN: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_and_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_offset:
; GCN: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64_offset:
; GCN: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_and_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64_offset:
; GCN: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64:
; GCN: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_and_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret:
; GCN: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64:
; GCN: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_and_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64:
; GCN: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_and_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile and i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_offset:
; GCN: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_sub_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_offset:
; GCN: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64_offset:
; GCN: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_sub_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64_offset:
; GCN: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64:
; GCN: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_sub_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret:
; GCN: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64:
; GCN: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_sub_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64:
; GCN: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_sub_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile sub i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_offset:
; GCN: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_max_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_offset:
; GCN: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64_offset:
; GCN: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_max_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64_offset:
; GCN: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64:
; GCN: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_max_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret:
; GCN: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64:
; GCN: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_max_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64:
; GCN: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_max_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile max i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_offset:
; GCN: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umax_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_offset:
; GCN: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64_offset:
; GCN: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umax_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64_offset:
; GCN: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64:
; GCN: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umax_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret:
; GCN: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64:
; GCN: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umax_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64:
; GCN: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umax_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile umax i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_offset:
; GCN: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_min_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_offset:
; GCN: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64_offset:
; GCN: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_min_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64_offset:
; GCN: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64:
; GCN: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_min_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret:
; GCN: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64:
; GCN: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_min_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64:
; GCN: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_min_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile min i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_offset:
; GCN: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umin_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_offset:
; GCN: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64_offset:
; GCN: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umin_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64_offset:
; GCN: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64:
; GCN: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umin_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret:
; GCN: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64:
; GCN: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_umin_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64:
; GCN: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_umin_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile umin i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_offset:
; GCN: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_or_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_offset:
; GCN: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64_offset:
; GCN: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_or_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64_offset:
; GCN: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64:
; GCN: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_or_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret:
; GCN: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64:
; GCN: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_or_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64:
; GCN: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_or_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile or i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_offset:
; GCN: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xchg_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_offset:
; GCN: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64_offset:
; GCN: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xchg_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64_offset:
; GCN: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64:
; GCN: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xchg_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret:
; GCN: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64:
; GCN: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xchg_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64:
; GCN: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]],  v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xchg_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile xchg i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_offset:
; GCN: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xor_i64_offset(i64 addrspace(4)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_offset:
; GCN: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64_offset:
; GCN: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xor_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64_offset:
; GCN: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64:
; GCN: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xor_i64(i64 addrspace(4)* %out, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret:
; GCN: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in) {
entry:
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64:
; GCN: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @atomic_xor_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64:
; GCN: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_xor_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %tmp0 = atomicrmw volatile xor i64 addrspace(4)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_offset:
; GCN: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i64_offset(i64 addrspace(4)* %in, i64 addrspace(4)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %in, i64 4
  %val = load atomic i64, i64 addrspace(4)* %gep  seq_cst, align 8
  store i64 %val, i64 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64:
; GCN: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i64(i64 addrspace(4)* %in, i64 addrspace(4)* %out) {
entry:
  %val = load atomic i64, i64 addrspace(4)* %in seq_cst, align 8
  store i64 %val, i64 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_addr64_offset:
; GCN: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i64_addr64_offset(i64 addrspace(4)* %in, i64 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %in, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %val = load atomic i64, i64 addrspace(4)* %gep seq_cst, align 8
  store i64 %val, i64 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i64_addr64:
; GCN: flat_load_dwordx2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @atomic_load_i64_addr64(i64 addrspace(4)* %in, i64 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %in, i64 %index
  %val = load atomic i64, i64 addrspace(4)* %ptr seq_cst, align 8
  store i64 %val, i64 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_offset:
; GCN: flat_store_dwordx2 [[RET:v\[[0-9]+:[0-9]\]]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_offset(i64 %in, i64 addrspace(4)* %out) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  store atomic i64 %in, i64 addrspace(4)* %gep  seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64:
; GCN: flat_store_dwordx2 {{v\[[0-9]+:[0-9]\]}}, v[{{[0-9]+}}:{{[0-9]+}}]
define amdgpu_kernel void @atomic_store_i64(i64 %in, i64 addrspace(4)* %out) {
entry:
  store atomic i64 %in, i64 addrspace(4)* %out seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_addr64_offset:
; GCN: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64_offset(i64 %in, i64 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  store atomic i64 %in, i64 addrspace(4)* %gep seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i64_addr64:
; GCN: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_store_i64_addr64(i64 %in, i64 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  store atomic i64 %in, i64 addrspace(4)* %ptr seq_cst, align 8
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_offset:
; GCN: flat_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_offset(i64 addrspace(4)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(4)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_soffset:
; GCN: flat_atomic_cmpswap_x2 v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_soffset(i64 addrspace(4)* %out, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 9000
  %val = cmpxchg volatile i64 addrspace(4)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_offset:
; GCN: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]{{:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %old) {
entry:
  %gep = getelementptr i64, i64 addrspace(4)* %out, i64 4
  %val = cmpxchg volatile i64 addrspace(4)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_addr64_offset:
; GCN: flat_atomic_cmpswap_x2 v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64_offset(i64 addrspace(4)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(4)* %gep, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64_offset:
; GCN: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64_offset(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(4)* %ptr, i64 4
  %val = cmpxchg volatile i64 addrspace(4)* %gep, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64:
; GCN: flat_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64(i64 addrspace(4)* %out, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(4)* %out, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret:
; GCN: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %old) {
entry:
  %val = cmpxchg volatile i64 addrspace(4)* %out, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_addr64:
; GCN: flat_atomic_cmpswap_x2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
define amdgpu_kernel void @atomic_cmpxchg_i64_addr64(i64 addrspace(4)* %out, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(4)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i64_ret_addr64:
; GCN: flat_atomic_cmpswap_x2 v{{\[}}[[RET:[0-9]+]]:{{[0-9]+\]}}, v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dwordx2 {{v\[[0-9]+:[0-9]+\]}}, v{{\[}}[[RET]]:
define amdgpu_kernel void @atomic_cmpxchg_i64_ret_addr64(i64 addrspace(4)* %out, i64 addrspace(4)* %out2, i64 %in, i64 %index, i64 %old) {
entry:
  %ptr = getelementptr i64, i64 addrspace(4)* %out, i64 %index
  %val = cmpxchg volatile i64 addrspace(4)* %ptr, i64 %old, i64 %in seq_cst seq_cst
  %extract0 = extractvalue { i64, i1 } %val, 0
  store i64 %extract0, i64 addrspace(4)* %out2
  ret void
}

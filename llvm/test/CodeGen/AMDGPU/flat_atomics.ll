; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}atomic_add_i32_offset:
; GCN: flat_atomic_add v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_add_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile add i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_offset:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_add_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile add i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64_offset:
; GCN: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_add_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile add i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64_offset:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_add_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile add i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32:
; GCN: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_add_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile add i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_add_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile add i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_addr64:
; GCN: flat_atomic_add v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_add_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile add i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i32_ret_addr64:
; GCN: flat_atomic_add [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_add_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile add i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_offset:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_and_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile and i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_offset:
; GCN: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_and_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile and i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64_offset:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_and_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile and i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64_offset:
; GCN: flat_atomic_and [[RET:v[0-9]]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_and_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile and i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_and_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile and i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret:
; GCN: flat_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_and_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile and i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_addr64:
; GCN: flat_atomic_and v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_and_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile and i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i32_ret_addr64:
; GCN: flat_atomic_and [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_and_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile and i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_offset:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_sub_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile sub i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_offset:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_sub_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile sub i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64_offset:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_sub_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile sub i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64_offset:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_sub_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile sub i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_sub_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_sub_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile sub i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_addr64:
; GCN: flat_atomic_sub v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_sub_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile sub i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i32_ret_addr64:
; GCN: flat_atomic_sub [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_sub_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile sub i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_offset:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_max_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile max i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_offset:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_max_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile max i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64_offset:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_max_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile max i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64_offset:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_max_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile max i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_max_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile max i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_max_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile max i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_addr64:
; GCN: flat_atomic_smax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_max_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile max i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i32_ret_addr64:
; GCN: flat_atomic_smax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_max_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile max i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_offset:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umax_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile umax i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_offset:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umax_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile umax i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64_offset:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umax_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile umax i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64_offset:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umax_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile umax i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umax_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umax_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umax i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_addr64:
; GCN: flat_atomic_umax v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umax_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile umax i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i32_ret_addr64:
; GCN: flat_atomic_umax [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umax_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile umax i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_offset:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_min_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile min i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_offset:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_min_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile min i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64_offset:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_min_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile min i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64_offset:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_min_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile min i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_min_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile min i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_min_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile min i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_addr64:
; GCN: flat_atomic_smin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_min_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile min i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i32_ret_addr64:
; GCN: flat_atomic_smin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_min_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile min i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_offset:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umin_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile umin i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_offset:
; GCN: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umin_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile umin i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64_offset:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umin_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile umin i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64_offset:
; GCN: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umin_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile umin i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umin_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret:
; GCN: flat_atomic_umin v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_umin_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile umin i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_addr64:
; GCN: flat_atomic_umin v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_umin_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile umin i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i32_ret_addr64:
; GCN: flat_atomic_umin [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]{{$}}
  define void @atomic_umin_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile umin i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_offset:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_or_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile or i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_offset:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_or_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile or i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64_offset:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_or_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile or i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64_offset:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_or_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile or i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_or_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile or i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_or_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile or i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_addr64:
; GCN: flat_atomic_or v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_or_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile or i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i32_ret_addr64:
; GCN: flat_atomic_or [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_or_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile or i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_offset:
; GCN: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_xchg_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile xchg i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_offset:
; GCN: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xchg_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile xchg i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64_offset:
; GCN: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_xchg_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile xchg i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64_offset:
; GCN: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xchg_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile xchg i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32:
; GCN: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_xchg_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret:
; GCN: flat_atomic_swap [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xchg_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_addr64:
; GCN: flat_atomic_swap v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_xchg_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile xchg i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i32_ret_addr64:
; GCN: flat_atomic_swap [[RET:v[0-9]+]],  v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xchg_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile xchg i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; CMP_SWAP

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_offset:
; GCN: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define void @atomic_cmpxchg_i32_offset(i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_offset:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define void @atomic_cmpxchg_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64_offset:
; GCN: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define void @atomic_cmpxchg_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val  = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64_offset:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define void @atomic_cmpxchg_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val  = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32:
; GCN: flat_atomic_cmpswap v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
define void @atomic_cmpxchg_i32(i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32 addrspace(4)* %out, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}] glc
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define void @atomic_cmpxchg_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i32 %old) {
entry:
  %val = cmpxchg volatile i32 addrspace(4)* %out, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_addr64:
; GCN: flat_atomic_cmpswap v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]{{$}}
define void @atomic_cmpxchg_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = cmpxchg volatile i32 addrspace(4)* %ptr, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_cmpxchg_i32_ret_addr64:
; GCN: flat_atomic_cmpswap v[[RET:[0-9]+]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v[[RET]]
define void @atomic_cmpxchg_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index, i32 %old) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = cmpxchg volatile i32 addrspace(4)* %ptr, i32 %old, i32 %in seq_cst seq_cst
  %flag = extractvalue { i32, i1 } %val, 0
  store i32 %flag, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_offset:
; GCN: flat_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_xor_i32_offset(i32 addrspace(4)* %out, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile xor i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_offset:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xor_i32_ret_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = atomicrmw volatile xor i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64_offset:
; GCN: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_xor_i32_addr64_offset(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile xor i32 addrspace(4)* %gep, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64_offset:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xor_i32_ret_addr64_offset(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = atomicrmw volatile xor i32 addrspace(4)* %gep, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32:
; GCN: flat_atomic_xor v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
define void @atomic_xor_i32(i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xor_i32_ret(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in) {
entry:
  %val = atomicrmw volatile xor i32 addrspace(4)* %out, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_addr64:
; GCN: flat_atomic_xor v[{{[0-9]+:[0-9]+}}], v{{[0-9]+$}}
define void @atomic_xor_i32_addr64(i32 addrspace(4)* %out, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile xor i32 addrspace(4)* %ptr, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i32_ret_addr64:
; GCN: flat_atomic_xor [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_xor_i32_ret_addr64(i32 addrspace(4)* %out, i32 addrspace(4)* %out2, i32 %in, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %val = atomicrmw volatile xor i32 addrspace(4)* %ptr, i32 %in seq_cst
  store i32 %val, i32 addrspace(4)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_offset:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_load_i32_offset(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %in, i32 4
  %val = load atomic i32, i32 addrspace(4)* %gep  seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_load_i32(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64_offset:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_load_i32_addr64_offset(i32 addrspace(4)* %in, i32 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %in, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  %val = load atomic i32, i32 addrspace(4)* %gep seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_load_i32_addr64:
; GCN: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+:[0-9]+}}] glc{{$}}
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @atomic_load_i32_addr64(i32 addrspace(4)* %in, i32 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %in, i64 %index
  %val = load atomic i32, i32 addrspace(4)* %ptr seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_offset:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @atomic_store_i32_offset(i32 %in, i32 addrspace(4)* %out) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  store atomic i32 %in, i32 addrspace(4)* %gep  seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @atomic_store_i32(i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64_offset:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @atomic_store_i32_addr64_offset(i32 %in, i32 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  %gep = getelementptr i32, i32 addrspace(4)* %ptr, i32 4
  store atomic i32 %in, i32 addrspace(4)* %gep seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}atomic_store_i32_addr64:
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @atomic_store_i32_addr64(i32 %in, i32 addrspace(4)* %out, i64 %index) {
entry:
  %ptr = getelementptr i32, i32 addrspace(4)* %out, i64 %index
  store atomic i32 %in, i32 addrspace(4)* %ptr seq_cst, align 4
  ret void
}

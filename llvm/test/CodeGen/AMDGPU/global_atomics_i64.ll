; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s


; GCN-LABEL: {{^}}atomic_add_i64_offset:
; GCN: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_add_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_offset:
; GCN: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_add_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64_offset:
; CI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}}{{$}}
define void @atomic_add_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64_offset:
; CI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_add_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64:
; GCN: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_add_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret:
; GCN: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_add_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_addr64:
; CI: buffer_atomic_add_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_add_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_add_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_add_i64_ret_addr64:
; CI: buffer_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_add_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_add_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile add i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_offset:
; GCN: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_and_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_offset:
; GCN: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_and_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64_offset:
; CI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_and_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64_offset:
; CI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_and_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64:
; GCN: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_and_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret:
; GCN: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_and_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_addr64:
; CI: buffer_atomic_and_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_and_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_and_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_and_i64_ret_addr64:
; CI: buffer_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_and_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_and_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile and i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_offset:
; GCN: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_sub_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_offset:
; GCN: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_sub_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64_offset:
; CI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_sub_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64_offset:
; CI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_sub_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64:
; GCN: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_sub_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret:
; GCN: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_sub_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_addr64:
; CI: buffer_atomic_sub_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_sub_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_sub_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_sub_i64_ret_addr64:
; CI: buffer_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_sub_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_sub_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile sub i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_offset:
; GCN: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_max_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_offset:
; GCN: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_max_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64_offset:
; CI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_max_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64_offset:
; CI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_max_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64:
; GCN: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_max_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret:
; GCN: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_max_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_addr64:
; CI: buffer_atomic_smax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_max_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_max_i64_ret_addr64:
; CI: buffer_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_max_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile max i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_offset:
; GCN: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_umax_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_offset:
; GCN: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umax_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64_offset:
; CI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_umax_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64_offset:
; CI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umax_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64:
; GCN: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_umax_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret:
; GCN: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umax_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_addr64:
; CI: buffer_atomic_umax_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umax_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_umax_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umax_i64_ret_addr64:
; CI: buffer_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umax_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umax_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile umax i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_offset:
; GCN: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_min_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_offset:
; GCN: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_min_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64_offset:
; CI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_min_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64_offset:
; CI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_min_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64:
; GCN: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_min_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret:
; GCN: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_min_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_addr64:
; CI: buffer_atomic_smin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_smin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_min_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_min_i64_ret_addr64:
; CI: buffer_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_smin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_min_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile min i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_offset:
; GCN: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_umin_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_offset:
; GCN: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umin_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64_offset:
; CI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_umin_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64_offset:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umin_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64:
; GCN: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_umin_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umin_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_addr64:
; CI: buffer_atomic_umin_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_umin_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_umin_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_umin_i64_ret_addr64:
; CI: buffer_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_umin_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_umin_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile umin i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_offset:
; GCN: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_or_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_offset:
; GCN: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_or_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64_offset:
; CI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_or_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64_offset:
; CI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_or_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64:
; GCN: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_or_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret:
; GCN: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_or_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_addr64:
; CI: buffer_atomic_or_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_or_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_or_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_or_i64_ret_addr64:
; CI: buffer_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_or_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_or_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile or i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_offset:
; GCN: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_xchg_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_offset:
; GCN: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xchg_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64_offset:
; CI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
define void @atomic_xchg_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64_offset:
; CI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xchg_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64:
; GCN: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_xchg_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret:
; GCN: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xchg_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_addr64:
; CI: buffer_atomic_swap_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_swap_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_xchg_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xchg_i64_ret_addr64:
; CI: buffer_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_swap_x2 [[RET:v\[[0-9]+:[0-9]+\]]],  v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xchg_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile xchg i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_offset:
; GCN: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32{{$}}
define void @atomic_xor_i64_offset(i64 addrspace(1)* %out, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_offset:
; GCN: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 offset:32 glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xor_i64_ret_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %gep = getelementptr i64, i64 addrspace(1)* %out, i64 4
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64_offset:
; CI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32{{$}}
; VI: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_xor_i64_addr64_offset(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64_offset:
; CI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 offset:32 glc{{$}}
; VI: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xor_i64_ret_addr64_offset(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %gep = getelementptr i64, i64 addrspace(1)* %ptr, i64 4
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %gep, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64:
; GCN: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0{{$}}
define void @atomic_xor_i64(i64 addrspace(1)* %out, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret:
; GCN: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xor_i64_ret(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in) {
entry:
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %out, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_addr64:
; CI: buffer_atomic_xor_x2 v{{\[[0-9]+:[0-9]+\]}}, v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64{{$}}
; VI: flat_atomic_xor_x2 v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]$}}
define void @atomic_xor_i64_addr64(i64 addrspace(1)* %out, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}atomic_xor_i64_ret_addr64:
; CI: buffer_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+}}:{{[0-9]+}}], s[{{[0-9]+}}:{{[0-9]+}}], 0 addr64 glc{{$}}
; VI: flat_atomic_xor_x2 [[RET:v\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v{{\[[0-9]+:[0-9]+\]}} glc{{$}}
; GCN: buffer_store_dwordx2 [[RET]]
define void @atomic_xor_i64_ret_addr64(i64 addrspace(1)* %out, i64 addrspace(1)* %out2, i64 %in, i64 %index) {
entry:
  %ptr = getelementptr i64, i64 addrspace(1)* %out, i64 %index
  %tmp0  = atomicrmw volatile xor i64 addrspace(1)* %ptr, i64 %in seq_cst
  store i64 %tmp0, i64 addrspace(1)* %out2
  ret void
}

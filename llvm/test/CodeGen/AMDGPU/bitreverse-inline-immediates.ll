; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Test that materialization constants that are the bit reversed of
; inline immediates are replaced with bfrev of the inline immediate to
; save code size.

; GCN-LABEL: {{^}}materialize_0_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_0_i32(ptr addrspace(1) %out) {
  store i32 0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_0_i64:
; GCN: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[HIK:[0-9]+]], v[[LOK]]{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_0_i64(ptr addrspace(1) %out) {
  store i64 0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_neg1_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], -1{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_neg1_i32(ptr addrspace(1) %out) {
  store i32 -1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_neg1_i64:
; GCN: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN: v_mov_b32_e32 v[[HIK:[0-9]+]], v[[LOK]]{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_neg1_i64(ptr addrspace(1) %out) {
  store i64 -1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_signbit_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_signbit_i32(ptr addrspace(1) %out) {
  store i32 -2147483648, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_signbit_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], 1{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_signbit_i64(ptr addrspace(1) %out) {
  store i64  -9223372036854775808, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg16_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], -16{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_neg16_i32(ptr addrspace(1) %out) {
  store i32 268435455, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg16_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], -16{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_neg16_i64(ptr addrspace(1) %out) {
  store i64  1152921504606846975, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg17_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0xf7ffffff{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_neg17_i32(ptr addrspace(1) %out) {
  store i32 -134217729, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg17_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0xf7ffffff{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_neg17_i64(ptr addrspace(1) %out) {
  store i64 -576460752303423489, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_64_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 64{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_64_i32(ptr addrspace(1) %out) {
  store i32 33554432, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_64_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], 64{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_64_i64(ptr addrspace(1) %out) {
  store i64 144115188075855872, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_65_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x82000000{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_65_i32(ptr addrspace(1) %out) {
  store i32 -2113929216, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_65_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0x82000000{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_65_i64(ptr addrspace(1) %out) {
  store i64 -9079256848778919936, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_3_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], -2.0{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_3_i32(ptr addrspace(1) %out) {
  store i32 -1073741824, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_3_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], -2.0{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_3_i64(ptr addrspace(1) %out) {
  store i64 -4611686018427387904, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_1.0_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x1fc{{$}}
; GCN: buffer_store_dword [[K]]
define amdgpu_kernel void @materialize_rev_1.0_i32(ptr addrspace(1) %out) {
  store i32 508, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_1.0_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0x1fc{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0{{$}}
; GCN: buffer_store_dwordx2 v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_1.0_i64(ptr addrspace(1) %out) {
  store i64 508, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}s_materialize_0_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0{{$}}
define amdgpu_kernel void @s_materialize_0_i32() {
  call void asm sideeffect "; use $0", "s"(i32 0)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_1_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 1{{$}}
define amdgpu_kernel void @s_materialize_1_i32() {
  call void asm sideeffect "; use $0", "s"(i32 1)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_neg1_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, -1{{$}}
define amdgpu_kernel void @s_materialize_neg1_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -1)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_signbit_i32:
; GCN: s_brev_b32 s{{[0-9]+}}, 1{{$}}
define amdgpu_kernel void @s_materialize_signbit_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -2147483648)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_rev_64_i32:
; GCN: s_brev_b32 s{{[0-9]+}}, 64{{$}}
define amdgpu_kernel void @s_materialize_rev_64_i32() {
  call void asm sideeffect "; use $0", "s"(i32 33554432)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_rev_65_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0x82000000{{$}}
define amdgpu_kernel void @s_materialize_rev_65_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -2113929216)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_rev_neg16_i32:
; GCN: s_brev_b32 s{{[0-9]+}}, -16{{$}}
define amdgpu_kernel void @s_materialize_rev_neg16_i32() {
  call void asm sideeffect "; use $0", "s"(i32 268435455)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_rev_neg17_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0xf7ffffff{{$}}
define amdgpu_kernel void @s_materialize_rev_neg17_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -134217729)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_rev_1.0_i32:
; GCN: s_movk_i32 s{{[0-9]+}}, 0x1fc{{$}}
define amdgpu_kernel void @s_materialize_rev_1.0_i32() {
  call void asm sideeffect "; use $0", "s"(i32 508)
  ret void
}

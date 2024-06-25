; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11 %s

; Test that materialization constants that are the bit reversed of
; inline immediates are replaced with bfrev of the inline immediate to
; save code size.

; GCN-LABEL: {{^}}materialize_0_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_0_i32(ptr addrspace(1) %out) {
  store i32 0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_0_i64:
; GCN: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 v[[HIK:[0-9]+]], v[[LOK]]{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_0_i64(ptr addrspace(1) %out) {
  store i64 0, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_neg1_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], -1{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_neg1_i32(ptr addrspace(1) %out) {
  store i32 -1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_neg1_i64:
; GCN: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN: v_mov_b32_e32 v[[HIK:[0-9]+]], v[[LOK]]{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_neg1_i64(ptr addrspace(1) %out) {
  store i64 -1, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_signbit_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 1{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_signbit_i32(ptr addrspace(1) %out) {
  store i32 -2147483648, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_signbit_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], 1{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_signbit_i64(ptr addrspace(1) %out) {
  store i64  -9223372036854775808, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg16_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], -16{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg16_i32(ptr addrspace(1) %out) {
  store i32 268435455, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg16_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], -16{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_neg16_i64(ptr addrspace(1) %out) {
  store i64  1152921504606846975, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg17_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0xf7ffffff{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg17_i32(ptr addrspace(1) %out) {
  store i32 -134217729, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg17_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], -1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0xf7ffffff{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_neg17_i64(ptr addrspace(1) %out) {
  store i64 -576460752303423489, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_64_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 64{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_64_i32(ptr addrspace(1) %out) {
  store i32 33554432, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_64_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HIK:[0-9]+]], 64{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_64_i64(ptr addrspace(1) %out) {
  store i64 144115188075855872, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_65_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x82000000{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_65_i32(ptr addrspace(1) %out) {
  store i32 -2113929216, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_65_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0x82000000{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_65_i64(ptr addrspace(1) %out) {
  store i64 -9079256848778919936, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_3_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], -2.0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_3_i32(ptr addrspace(1) %out) {
  store i32 -1073741824, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_3_i64:
; GCN-DAG: v_mov_b32_e32 v[[LOK:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], -2.0{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
define amdgpu_kernel void @materialize_rev_3_i64(ptr addrspace(1) %out) {
  store i64 -4611686018427387904, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_0.5_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 0.5{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_0.5_i32(ptr addrspace(1) %out) {
  store i32 252, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_1.0_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 1.0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_1.0_i32(ptr addrspace(1) %out) {
  store i32 508, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_2.0_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 2{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_2.0_i32(ptr addrspace(1) %out) {
  store i32 2, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_4.0_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], 4.0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_4.0_i32(ptr addrspace(1) %out) {
  store i32 258, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg0.5_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], -0.5{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg0.5_i32(ptr addrspace(1) %out) {
  store i32 253, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg1.0_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], -1.0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg1.0_i32(ptr addrspace(1) %out) {
  store i32 509, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg2.0_i32:
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 3{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg2.0_i32(ptr addrspace(1) %out) {
  store i32 3, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_neg4.0_i32:
; GCN: v_bfrev_b32_e32 [[K:v[0-9]+]], -4.0{{$}}
; GCN: {{buffer|flat}}_store_dword {{.*}}[[K]]
define amdgpu_kernel void @materialize_rev_neg4.0_i32(ptr addrspace(1) %out) {
  store i32 259, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_rev_1.0_i64:
; GCN-DAG: v_bfrev_b32_e32 v[[LOK:[0-9]+]], 1.0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIK:[0-9]+]], 0{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v[[[LOK]]:[[HIK]]]
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

; GCN-LABEL: {{^}}s_materialize_not_1.0_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0xc07fffff
define void @s_materialize_not_1.0_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -1065353217)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_not_neg_1.0_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0x407fffff
define void @s_materialize_not_neg_1.0_i32() {
  call void asm sideeffect "; use $0", "s"(i32 1082130431)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_not_inv2pi_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0xc1dd067c
define void @s_materialize_not_inv2pi_i32() {
  call void asm sideeffect "; use $0", "s"(i32 -1042479492)
  ret void
}

; GCN-LABEL: {{^}}s_materialize_not_neg_inv2pi_i32:
; GCN: s_mov_b32 s{{[0-9]+}}, 0x41dd067c
define void @s_materialize_not_neg_inv2pi_i32() {
  call void asm sideeffect "; use $0", "s"(i32 1105004156)
  ret void
}

; GCN-LABEL: {{^}}materialize_not_0.5_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, 0.5
define void @materialize_not_0.5_i32(ptr addrspace(1) %out) {
  store i32 -1056964609, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_1.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, 1.0
define void @materialize_not_1.0_i32(ptr addrspace(1) %out) {
  store i32 -1065353217, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_2.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, 2.0
define void @materialize_not_2.0_i32(ptr addrspace(1) %out) {
  store i32 -1073741825, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_4.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, 4.0
define void @materialize_not_4.0_i32(ptr addrspace(1) %out) {
  store i32 -1082130433, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_neg_0.5_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, -0.5
define void @materialize_not_neg_0.5_i32(ptr addrspace(1) %out) {
  store i32 1090519039, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_neg_1.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, -1.0
define void @materialize_not_neg_1.0_i32(ptr addrspace(1) %out) {
  store i32 1082130431, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_neg2.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, -2.0
define void @materialize_not_neg2.0_i32(ptr addrspace(1) %out) {
  store i32 1073741823, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_neg4.0_i32:
; GCN: v_not_b32_e32 v{{[0-9]+}}, -4.0
define void @materialize_not_neg4.0_i32(ptr addrspace(1) %out) {
  store i32 1065353215, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_inv2pi_i32:
; SI: v_mov_b32_e32 v{{[0-9]+}}, 0xc1dd067c
; VI: v_not_b32_e32 v{{[0-9]+}}, 0.15915494
define void @materialize_not_inv2pi_i32(ptr addrspace(1) %out) {
  store i32 -1042479492, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}materialize_not_neg_inv2pi_i32:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, 0x41dd067c
define void @materialize_not_neg_inv2pi_i32(ptr addrspace(1) %out) {
  store i32 1105004156, ptr addrspace(1) %out
  ret void
}

; One constant is reversible, the other is not. We shouldn't break
; vopd packing for this.
; GFX11-LABEL: {{^}}vopd_materialize:
; FIXME-GFX11: v_dual_mov_b32 v0, 0x102 :: v_dual_mov_b32 v1, 1.0
; GFX11: v_bfrev_b32_e32 v0, 4.0
; GFX11: v_mov_b32_e32 v1, 1.0
define <2 x i32> @vopd_materialize() {
  %insert0 = insertelement <2 x i32> poison, i32 258, i32 0
  %insert1 = insertelement <2 x i32> %insert0, i32 1065353216, i32 1
  ret <2 x i32> %insert1
}

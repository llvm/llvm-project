; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck -check-prefixes=GCN,FUNC %s
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck -check-prefixes=R600,FUNC %s

; BFI_INT Definition pattern from ISA docs
; (y & x) | (z & ~x)
;
; FUNC-LABEL: {{^}}bfi_def:
; R600: BFI_INT

; GCN:   v_bfi_b32
define amdgpu_kernel void @bfi_def(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = xor i32 %x, -1
  %1 = and i32 %z, %0
  %2 = and i32 %y, %x
  %3 = or i32 %1, %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; SHA-256 Ch function
; z ^ (x & (y ^ z))
; FUNC-LABEL: {{^}}bfi_sha256_ch:
; R600: BFI_INT

; GCN:   v_bfi_b32
define amdgpu_kernel void @bfi_sha256_ch(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = xor i32 %y, %z
  %1 = and i32 %x, %0
  %2 = xor i32 %z, %1
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; SHA-256 Ma function
; ((x & z) | (y & (x | z)))
; FUNC-LABEL: {{^}}bfi_sha256_ma:
; R600: XOR_INT * [[DST:T[0-9]+\.[XYZW]]], KC0[2].Z, KC0[2].W
; R600: BFI_INT * {{T[0-9]+\.[XYZW]}}, {{[[DST]]|PV\.[XYZW]}}, KC0[3].X, KC0[2].W

; GCN: v_xor_b32_e32 [[DST:v[0-9]+]], {{s[0-9]+, v[0-9]+}}
; GCN: v_bfi_b32 {{v[0-9]+}}, [[DST]], {{s[0-9]+, v[0-9]+}}
define amdgpu_kernel void @bfi_sha256_ma(i32 addrspace(1)* %out, i32 %x, i32 %y, i32 %z) {
entry:
  %0 = and i32 %x, %z
  %1 = or i32 %x, %z
  %2 = and i32 %y, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_bitselect_v2i32_pat1:
; GCN: s_waitcnt
; GCN-NEXT: v_bfi_b32 v1, v3, v1, v5
; GCN-NEXT: v_bfi_b32 v0, v2, v0, v4
; GCN-NEXT: s_setpc_b64
define <2 x i32> @v_bitselect_v2i32_pat1(<2 x i32> %a, <2 x i32> %b, <2 x i32> %mask) {
  %xor.0 = xor <2 x i32> %a, %mask
  %and = and <2 x i32> %xor.0, %b
  %bitselect = xor <2 x i32> %and, %mask
  ret <2 x i32> %bitselect
}

; FUNC-LABEL: {{^}}v_bitselect_i64_pat_0:
; GCN: s_waitcnt
; GCN-NEXT: v_bfi_b32 v1, v1, v3, v5
; GCN-NEXT: v_bfi_b32 v0, v0, v2, v4
; GCN-NEXT: s_setpc_b64
define i64 @v_bitselect_i64_pat_0(i64 %a, i64 %b, i64 %mask) {
  %and0 = and i64 %a, %b
  %not.a = xor i64 %a, -1
  %and1 = and i64 %not.a, %mask
  %bitselect = or i64 %and0, %and1
  ret i64 %bitselect
}

; FUNC-LABEL: {{^}}v_bitselect_i64_pat_1:
; GCN: s_waitcnt
; GCN-NEXT: v_bfi_b32 v1, v3, v1, v5
; GCN-NEXT: v_bfi_b32 v0, v2, v0, v4
; GCN-NEXT: s_setpc_b64
define i64 @v_bitselect_i64_pat_1(i64 %a, i64 %b, i64 %mask) {
  %xor.0 = xor i64 %a, %mask
  %and = and i64 %xor.0, %b
  %bitselect = xor i64 %and, %mask
  ret i64 %bitselect
}

; FUNC-LABEL: {{^}}v_bitselect_i64_pat_2:
; GCN: s_waitcnt
; GCN-DAG: v_bfi_b32 v0, v2, v0, v4
; GCN-DAG: v_bfi_b32 v1, v3, v1, v5
; GCN-NEXT: s_setpc_b64
define i64 @v_bitselect_i64_pat_2(i64 %a, i64 %b, i64 %mask) {
  %xor.0 = xor i64 %a, %mask
  %and = and i64 %xor.0, %b
  %bitselect = xor i64 %and, %mask
  ret i64 %bitselect
}

; FUNC-LABEL: {{^}}v_bfi_sha256_ma_i64:
; GCN-DAG: v_xor_b32_e32 v1, v1, v3
; GCN-DAG: v_xor_b32_e32 v0, v0, v2
; GCN-DAG: v_bfi_b32 v1, v1, v5, v3
; GCN-DAG: v_bfi_b32 v0, v0, v4, v2
define i64 @v_bfi_sha256_ma_i64(i64 %x, i64 %y, i64 %z) {
entry:
  %and0 = and i64 %x, %z
  %or0 = or i64 %x, %z
  %and1 = and i64 %y, %or0
  %or1 = or i64 %and0, %and1
  ret i64 %or1
}

; FIXME: Should leave as 64-bit SALU ops
; FUNC-LABEL: {{^}}s_bitselect_i64_pat_0:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_bfi_b32
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_bfi_b32
define amdgpu_kernel void @s_bitselect_i64_pat_0(i64 %a, i64 %b, i64 %mask) {
  %and0 = and i64 %a, %b
  %not.a = xor i64 %a, -1
  %and1 = and i64 %not.a, %mask
  %bitselect = or i64 %and0, %and1
  %scalar.use = add i64 %bitselect, 10
  store i64 %scalar.use, i64 addrspace(1)* undef
  ret void
}

; FUNC-LABEL: {{^}}s_bitselect_i64_pat_1:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_bfi_b32
define amdgpu_kernel void @s_bitselect_i64_pat_1(i64 %a, i64 %b, i64 %mask) {
  %xor.0 = xor i64 %a, %mask
  %and = and i64 %xor.0, %b
  %bitselect = xor i64 %and, %mask

  %scalar.use = add i64 %bitselect, 10
  store i64 %scalar.use, i64 addrspace(1)* undef
  ret void
}

; FUNC-LABEL: {{^}}s_bitselect_i64_pat_2:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN-DAG: v_bfi_b32
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN: v_bfi_b32
define amdgpu_kernel void @s_bitselect_i64_pat_2(i64 %a, i64 %b, i64 %mask) {
  %xor.0 = xor i64 %a, %mask
  %and = and i64 %xor.0, %b
  %bitselect = xor i64 %and, %mask

  %scalar.use = add i64 %bitselect, 10
  store i64 %scalar.use, i64 addrspace(1)* undef
  ret void
}

; FUNC-LABEL: {{^}}s_bfi_sha256_ma_i64:
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN-DAG: v_xor_b32
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, s
; GCN-DAG: v_xor_b32
; GCN-DAG: v_bfi_b32
; GCN: v_bfi_b32
define amdgpu_kernel void @s_bfi_sha256_ma_i64(i64 %x, i64 %y, i64 %z) {
entry:
  %and0 = and i64 %x, %z
  %or0 = or i64 %x, %z
  %and1 = and i64 %y, %or0
  %or1 = or i64 %and0, %and1

  %scalar.use = add i64 %or1, 10
  store i64 %scalar.use, i64 addrspace(1)* undef
  ret void
}

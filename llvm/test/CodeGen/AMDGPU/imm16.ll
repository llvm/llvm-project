; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=tonga -mattr=-flat-for-global -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s

; FIXME: Merge into imm.ll

; GCN-LABEL: {{^}}store_inline_imm_neg_0.0_i16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x8000{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff8000{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_neg_0.0_i16(i16 addrspace(1)* %out) {
  store volatile i16 -32768, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_0.0_f16(half addrspace(1)* %out) {
  store half 0.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_imm_neg_0.0_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x8000{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffff8000{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_imm_neg_0.0_f16(half addrspace(1)* %out) {
  store half -0.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.5_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3800{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_0.5_f16(half addrspace(1)* %out) {
  store half 0.5, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_0.5_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb800{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffb800{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_m_0.5_f16(half addrspace(1)* %out) {
  store half -0.5, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_1.0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_1.0_f16(half addrspace(1)* %out) {
  store half 1.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_1.0_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbc00{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffbc00{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_m_1.0_f16(half addrspace(1)* %out) {
  store half -1.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_2.0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4000{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_2.0_f16(half addrspace(1)* %out) {
  store half 2.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_2.0_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc000{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffc000{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_m_2.0_f16(half addrspace(1)* %out) {
  store half -2.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_4.0_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x4400{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_4.0_f16(half addrspace(1)* %out) {
  store half 4.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_4.0_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc400{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffc400{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_m_4.0_f16(half addrspace(1)* %out) {
  store half -4.0, half addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}store_inline_imm_inv_2pi_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3118{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_inv_2pi_f16(half addrspace(1)* %out) {
  store half 0xH3118, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_inv_2pi_f16:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb118{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0xffffb118{{$}}
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_inline_imm_m_inv_2pi_f16(half addrspace(1)* %out) {
  store half 0xHB118, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_literal_imm_f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x6c00
; GCN: buffer_store_short [[REG]]
define amdgpu_kernel void @store_literal_imm_f16(half addrspace(1)* %out) {
  store half 4096.0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_0.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.5_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 0.5{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_0.5_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0.5
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_0.5_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], -0.5{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_0.5_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, -0.5
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 1.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_1.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 1.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], -1.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, -1.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 2.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_2.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 2.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], -2.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, -2.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_4.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 4.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_4.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 4.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_4.0_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], -4.0{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_4.0_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, -4.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_inline_imm_0.5_f16:
; VI: buffer_load_ushort [[VAL:v[0-9]+]]
; VI: v_add_f16_e32 [[REG:v[0-9]+]], 0.5, [[VAL]]
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @commute_add_inline_imm_0.5_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %x = load half, half addrspace(1)* %in
  %y = fadd half %x, 0.5
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_literal_f16:
; VI: buffer_load_ushort [[VAL:v[0-9]+]]
; VI: v_add_f16_e32 [[REG:v[0-9]+]], 0x6400, [[VAL]]
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @commute_add_literal_f16(half addrspace(1)* %out, half addrspace(1)* %in) {
  %x = load half, half addrspace(1)* %in
  %y = fadd half %x, 1024.0
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 1{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_1_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0xH0001
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 2{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_2_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0xH0002
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_16_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 16{{$}}
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_16_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0xH0010
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1_f16:
; VI: v_add_u32_e32 [[REG:v[0-9]+]], vcc, -1
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1_f16(half addrspace(1)* %out, i16 addrspace(1)* %in) {
  %x = load i16, i16 addrspace(1)* %in
  %y = add i16 %x, -1
  %ybc = bitcast i16 %y to half
  store half %ybc, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2_f16:
; VI: v_add_u32_e32 [[REG:v[0-9]+]], vcc, 0xfffe
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2_f16(half addrspace(1)* %out, i16 addrspace(1)* %in) {
  %x = load i16, i16 addrspace(1)* %in
  %y = add i16 %x, -2
  %ybc = bitcast i16 %y to half
  store half %ybc, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_16_f16:
; VI: v_add_u32_e32 [[REG:v[0-9]+]], vcc, 0xfff0
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_16_f16(half addrspace(1)* %out, i16 addrspace(1)* %in) {
  %x = load i16, i16 addrspace(1)* %in
  %y = add i16 %x, -16
  %ybc = bitcast i16 %y to half
  store half %ybc, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_63_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 63
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_63_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0xH003F
  store half %y, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_64_f16:
; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: v_add_f16_e64 [[REG:v[0-9]+]], [[VAL]], 64
; VI: buffer_store_short [[REG]]
define amdgpu_kernel void @add_inline_imm_64_f16(half addrspace(1)* %out, half %x) {
  %y = fadd half %x, 0xH0040
  store half %y, half addrspace(1)* %out
  ret void
}

; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s

; Use a 64-bit value with lo bits that can be represented as an inline constant
; GCN-LABEL: {{^}}i64_imm_inline_lo:
; GCN: v_mov_b32_e32 v[[LO_VGPR:[0-9]+]], 5
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VGPR]]:
define amdgpu_kernel void @i64_imm_inline_lo(i64 addrspace(1) *%out) {
entry:
  store i64 1311768464867721221, i64 addrspace(1) *%out ; 0x1234567800000005
  ret void
}

; Use a 64-bit value with hi bits that can be represented as an inline constant
; GCN-LABEL: {{^}}i64_imm_inline_hi:
; GCN: v_mov_b32_e32 v[[HI_VGPR:[0-9]+]], 5
; GCN: buffer_store_dwordx2 v{{\[[0-9]+:}}[[HI_VGPR]]
define amdgpu_kernel void @i64_imm_inline_hi(i64 addrspace(1) *%out) {
entry:
  store i64 21780256376, i64 addrspace(1) *%out ; 0x0000000512345678
  ret void
}

; GCN-LABEL: {{^}}store_imm_neg_0.0_i64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HI_VREG:[0-9]+]], 1{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_imm_neg_0.0_i64(i64 addrspace(1) *%out) {
  store i64 -9223372036854775808, i64 addrspace(1) *%out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_neg_0.0_i32:
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_neg_0.0_i32(i32 addrspace(1)* %out) {
  store i32 -2147483648, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_0.0_f32(float addrspace(1)* %out) {
  store float 0.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_imm_neg_0.0_f32:
; GCN: v_bfrev_b32_e32 [[REG:v[0-9]+]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_imm_neg_0.0_f32(float addrspace(1)* %out) {
  store float -0.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.5_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0.5{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_0.5_f32(float addrspace(1)* %out) {
  store float 0.5, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_0.5_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], -0.5{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_0.5_f32(float addrspace(1)* %out) {
  store float -0.5, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_1.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_1.0_f32(float addrspace(1)* %out) {
  store float 1.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_1.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], -1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_1.0_f32(float addrspace(1)* %out) {
  store float -1.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_2.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 2.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_2.0_f32(float addrspace(1)* %out) {
  store float 2.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_2.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], -2.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_2.0_f32(float addrspace(1)* %out) {
  store float -2.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_4.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 4.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_4.0_f32(float addrspace(1)* %out) {
  store float 4.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_4.0_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], -4.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_4.0_f32(float addrspace(1)* %out) {
  store float -4.0, float addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}store_inline_imm_inv_2pi_f32:
; SI: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3e22f983{{$}}
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], 0.15915494{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_inv_2pi_f32(float addrspace(1)* %out) {
  store float 0x3FC45F3060000000, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_inv_2pi_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbe22f983{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_inv_2pi_f32(float addrspace(1)* %out) {
  store float 0xBFC45F3060000000, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_literal_imm_f32:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x45800000
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_literal_imm_f32(float addrspace(1)* %out) {
  store float 4096.0, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_0.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.5_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 0.5{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_0.5_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_0.5_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -0.5{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_0.5_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_1.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 1.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -1.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -1.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 2.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_2.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 2.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -2.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -2.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_4.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 4.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_4.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 4.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_4.0_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -4.0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_4.0_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, -4.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_inline_imm_0.5_f32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_add_f32_e32 [[REG:v[0-9]+]], 0.5, [[VAL]]
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_add_inline_imm_0.5_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %x = load float, float addrspace(1)* %in
  %y = fadd float %x, 0.5
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_literal_f32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]]
; GCN: v_add_f32_e32 [[REG:v[0-9]+]], 0x44800000, [[VAL]]
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_add_literal_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %x = load float, float addrspace(1)* %in
  %y = fadd float %x, 1024.0
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_1_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0x36a0000000000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 2{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_2_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0x36b0000000000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_16_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 16
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_16_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0x36e0000000000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -1{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0xffffffffe0000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -2{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0xffffffffc0000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_16_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], -16
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_16_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0xfffffffe00000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_63_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 63
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_63_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0x36ff800000000000
  store float %y, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_64_f32:
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN: v_add_f32_e64 [[REG:v[0-9]+]], [[VAL]], 64
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_64_f32(float addrspace(1)* %out, float %x) {
  %y = fadd float %x, 0x3700000000000000
  store float %y, float addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}add_inline_imm_0.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 0{{$}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_0.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.5_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 0.5
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_0.5_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0.5
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_0.5_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -0.5
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_0.5_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, -0.5
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 1.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_1.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 1.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -1.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, -1.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 2.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_2.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 2.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -2.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, -2.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_4.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 4.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_4.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 4.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_4.0_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -4.0
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_4.0_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, -4.0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_inv_2pi_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0x6dc9c882
; SI-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x3fc45f30
; SI: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}

; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; VI: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 0.15915494{{$}}
; VI: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_inv_2pi_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x3fc45f306dc9c882
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_m_inv_2pi_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0x6dc9c882
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0xbfc45f30
; GCN: v_add_f64 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @add_m_inv_2pi_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0xbfc45f306dc9c882
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 1{{$}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_1_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x0000000000000001
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 2{{$}}
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_2_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x0000000000000002
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_16_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 16
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_16_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x0000000000000010
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -1
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0xffffffffffffffff
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -2
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0xfffffffffffffffe
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_16_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], -16
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_16_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0xfffffffffffffff0
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_63_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 63
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_63_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x000000000000003F
  store double %y, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_64_f64:
; SI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xb
; VI: s_load_dwordx2 [[VAL:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN: v_add_f64 [[REG:v\[[0-9]+:[0-9]+\]]], [[VAL]], 64
; GCN: buffer_store_dwordx2 [[REG]]
define amdgpu_kernel void @add_inline_imm_64_f64(double addrspace(1)* %out, double %x) {
  %y = fadd double %x, 0x0000000000000040
  store double %y, double addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}store_inline_imm_0.0_f64:
; GCN: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0
; GCN: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], v[[LO_VREG]]{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_0.0_f64(double addrspace(1)* %out) {
  store double 0.0, double addrspace(1)* %out
  ret void
}


; GCN-LABEL: {{^}}store_literal_imm_neg_0.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_bfrev_b32_e32 v[[HI_VREG:[0-9]+]], 1{{$}}
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_literal_imm_neg_0.0_f64(double addrspace(1)* %out) {
  store double -0.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.5_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x3fe00000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_0.5_f64(double addrspace(1)* %out) {
  store double 0.5, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_0.5_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0xbfe00000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_m_0.5_f64(double addrspace(1)* %out) {
  store double -0.5, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_1.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x3ff00000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_1.0_f64(double addrspace(1)* %out) {
  store double 1.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_1.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0xbff00000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_m_1.0_f64(double addrspace(1)* %out) {
  store double -1.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_2.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 2.0
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_2.0_f64(double addrspace(1)* %out) {
  store double 2.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_2.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], -2.0
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_m_2.0_f64(double addrspace(1)* %out) {
  store double -2.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_4.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x40100000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_4.0_f64(double addrspace(1)* %out) {
  store double 4.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_4.0_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0xc0100000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_m_4.0_f64(double addrspace(1)* %out) {
  store double -4.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inv_2pi_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0x6dc9c882
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x3fc45f30
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inv_2pi_f64(double addrspace(1)* %out) {
  store double 0x3fc45f306dc9c882, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_inv_2pi_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0x6dc9c882
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0xbfc45f30
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_inline_imm_m_inv_2pi_f64(double addrspace(1)* %out) {
  store double 0xbfc45f306dc9c882, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_literal_imm_f64:
; GCN-DAG: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], 0x40b00000
; GCN: buffer_store_dwordx2 v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define amdgpu_kernel void @store_literal_imm_f64(double addrspace(1)* %out) {
  store double 4096.0, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}literal_folding:
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0x3f4353f8, v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, 0xbf4353f8, v{{[0-9]+}}
define amdgpu_vs void @literal_folding(float %arg) {
main_body:
  %tmp = fmul float %arg, 0x3FE86A7F00000000
  %tmp1 = fmul float %arg, 0xBFE86A7F00000000
  call void @llvm.amdgcn.exp.f32(i32 12, i32 15, float %tmp, float %tmp, float %tmp1, float %tmp1, i1 true, i1 false) #0
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #0

attributes #0 = { nounwind }

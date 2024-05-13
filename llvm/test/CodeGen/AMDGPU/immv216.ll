; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=gfx1100 -mattr=-flat-for-global -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr=-flat-for-global,-xnack -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=-flat-for-global,-xnack -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global,-xnack -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global -verify-machineinstrs -show-mc-encoding < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s
; FIXME: Merge into imm.ll

; GCN-LABEL: {{^}}store_inline_imm_neg_0.0_v2i16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_neg_0.0_v2i16(ptr addrspace(1) %out) #0 {
  store <2 x i16> <i16 -32768, i16 -32768>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_0.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 0.0, half 0.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_imm_neg_0.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_imm_neg_0.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half -0.0, half -0.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.5_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x38003800 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_0.5_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 0.5, half 0.5>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_0.5_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb800b800 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_m_0.5_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half -0.5, half -0.5>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_1.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c003c00 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_1.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 1.0, half 1.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_1.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbc00bc00 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_m_1.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half -1.0, half -1.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_2.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x40004000 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_2.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 2.0, half 2.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_2.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc000c000 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_m_2.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half -2.0, half -2.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_4.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x44004400 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_4.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 4.0, half 4.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_4.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc400c400 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_m_4.0_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half -4.0, half -4.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_inv_2pi_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x31183118 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_inv_2pi_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 0xH3118, half 0xH3118>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_inv_2pi_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb118b118 ; encoding
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_inline_imm_m_inv_2pi_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 0xHB118, half 0xHB118>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}store_literal_imm_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x6c006c00
; GCN: buffer_store_{{dword|b32}} [[REG]]
define amdgpu_kernel void @store_literal_imm_v2f16(ptr addrspace(1) %out) #0 {
  store <2 x half> <half 4096.0, half 4096.0>, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0 ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST0:v[0-9]+]], 0
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST0]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_0.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0.0, half 0.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.5_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5 op_sel_hi:[1,0] ; encoding: [0x00,0x40,0x0f,0xcc,0x02,0xe0,0x01,0x08]
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5 op_sel_hi:[1,0] ; encoding: [0x00,0x40,0x8f,0xd3,0x06,0xe0,0x01,0x08]
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST05:v[0-9]+]], 0x3800
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 0.5
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_0.5_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0.5, half 0.5>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_0.5_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -0.5 op_sel_hi:[1,0] ; encoding: [0x00,0x40,0x0f,0xcc,0x02,0xe2,0x01,0x08]
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -0.5 op_sel_hi:[1,0] ; encoding: [0x00,0x40,0x8f,0xd3,0x06,0xe2,0x01,0x08]
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONSTM05:v[0-9]+]], 0xb800
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONSTM05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], -0.5
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_0.5_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -0.5, half -0.5>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1.0 op_sel_hi:[1,0] ; encoding:
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST1:v[0-9]+]], 0x3c00
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 1.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_1.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 1.0, half 1.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -1.0 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -1.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]


; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST1:v[0-9]+]], 0xbc00
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], -1.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_1.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -1.0, half -1.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2.0 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 0x4000
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 2.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_2.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 2.0, half 2.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -2.0 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -2.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONSTM2:v[0-9]+]], 0xc000
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONSTM2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], -2.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_2.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -2.0, half -2.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_4.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 4.0 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 4.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST4:v[0-9]+]], 0x4400
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST4]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 4.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_4.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 4.0, half 4.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_4.0_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -4.0 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -4.0 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONSTM4:v[0-9]+]], 0xc400
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONSTM4]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], -4.0
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_4.0_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -4.0, half -4.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_inline_imm_0.5_v2f16:
; GFX10: buffer_load_{{dword|b32}} [[VAL:v[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: buffer_load_dword [[VAL:v[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5
; GFX9: buffer_store_dword [[REG]]

; VI-DAG: v_mov_b32_e32 [[CONST05:v[0-9]+]], 0x3800
; VI-DAG: buffer_load_dword
; VI-NOT: and
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[CONST05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 0.5, v{{[0-9]+}}
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @commute_add_inline_imm_0.5_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %x = load <2 x half>, ptr addrspace(1) %in
  %y = fadd <2 x half> %x, <half 0.5, half 0.5>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_literal_v2f16:
; GFX10: v_pk_add_f16 v0, 0x6400, v0 op_sel_hi:[0,1] ; encoding: [0x00,0x40,0x0f,0xcc,0xff,0x00,0x02,0x10,0x00,0x64,0x00,0x00]

; GFX9-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x6400 ; encoding
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], [[K]] op_sel_hi:[1,0] ; encoding: [0x00,0x40,0x8f,0xd3,0x00,0x01,0x00,0x08]
; GFX9: buffer_store_dword [[REG]]

; VI-DAG: buffer_load_dword
; VI-NOT: and
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 0x6400, v{{[0-9]+}}
; gfx8 does not support sreg or imm in sdwa - this will be move then
; VI-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x6400
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[VK]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: buffer_store_dword
define amdgpu_kernel void @commute_add_literal_v2f16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %x = load <2 x half>, ptr addrspace(1) %in
  %y = fadd <2 x half> %x, <half 1024.0, half 1024.0>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST1:v[0-9]+]], 1 ; encoding
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 1 ; encoding
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_1_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0001, half 0xH0001>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]


; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 2 ; encoding
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 2 ; encoding
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_2_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0002, half 0xH0002>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_16_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 16 op_sel_hi:[1,0] ; encoding
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 16 op_sel_hi:[1,0] ; encoding
; GFX9: buffer_store_dword [[REG]]


; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST16:v[0-9]+]], 16 ; encoding
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 16 ; encoding
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_16_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0010, half 0xH0010>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1_v2f16:
; GFX10: s_add_i32 [[VAL:s[0-9]+]], s{{[0-9]+}}, -1
; GFX10: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_add_i32 [[VAL:s[0-9]+]], s6, -1
; GFX9: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX9: buffer_store_dword [[REG]]

; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: s_add_i32 [[ADD:s[0-9]+]], [[VAL]], -1 ; encoding
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], [[ADD]]
; VI: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_1_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %xbc = bitcast <2 x half> %x to i32
  %y = add i32 %xbc, -1
  %ybc = bitcast i32 %y to <2 x half>
  store <2 x half> %ybc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2_v2f16:
; GFX10: s_add_i32 [[VAL:s[0-9]+]], s{{[0-9]+}}, 0xfffefffe
; GFX10: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_add_i32 [[VAL:s[0-9]+]], s6, 0xfffefffe
; GFX9: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX9: buffer_store_dword [[REG]]

; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: s_add_i32 [[ADD:s[0-9]+]], [[VAL]], 0xfffefffe ; encoding
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], [[ADD]]
; VI: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_2_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %xbc = bitcast <2 x half> %x to i32
  %y = add i32 %xbc, 4294901758 ; 0xfffefffe
  %ybc = bitcast i32 %y to <2 x half>
  store <2 x half> %ybc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_16_v2f16:
; GFX10: s_add_i32 [[VAL:s[0-9]+]], s{{[0-9]+}}, 0xfff0fff0
; GFX10: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_add_i32 [[VAL:s[0-9]+]], s6, 0xfff0fff0
; GFX9: v_mov_b32_e32 [[REG:v[0-9]+]], [[VAL]]
; GFX9: buffer_store_dword [[REG]]


; VI: s_load_dword [[VAL:s[0-9]+]]
; VI: s_add_i32 [[ADD:s[0-9]+]], [[VAL]], 0xfff0fff0 ; encoding
; VI: v_mov_b32_e32 [[REG:v[0-9]+]], [[ADD]]
; VI: buffer_store_dword [[REG]]
define amdgpu_kernel void @add_inline_imm_neg_16_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %xbc = bitcast <2 x half> %x to i32
  %y = add i32 %xbc, 4293984240 ; 0xfff0fff0
  %ybc = bitcast i32 %y to <2 x half>
  store <2 x half> %ybc, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_63_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 63
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 63
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST63:v[0-9]+]], 63
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST63]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 63
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_63_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH003F, half 0xH003F>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_64_v2f16:
; GFX10: s_load_{{dword|b32}} [[VAL:s[0-9]+]]
; GFX10: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 64
; GFX10: buffer_store_{{dword|b32}} [[REG]]

; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 64
; GFX9: buffer_store_dword [[REG]]

; FIXME: Shouldn't need right shift and SDWA, also extra copy
; VI-DAG: s_load_dword [[VAL:s[0-9]+]]
; VI-DAG: v_mov_b32_e32 [[CONST64:v[0-9]+]], 64
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[V_SHR:v[0-9]+]], [[SHR]]

; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[V_SHR]], [[CONST64]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI-DAG: v_add_f16_e64 v{{[0-9]+}}, [[VAL]], 64
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_64_v2f16(ptr addrspace(1) %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0040, half 0xH0040>
  store <2 x half> %y, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}mul_inline_imm_0.5_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0x3800
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0x3800, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0x38,0x00,0x00]
define <2 x i16> @mul_inline_imm_0.5_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half 0.5, half 0.5> to <2 x i16>)
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}mul_inline_imm_neg_0.5_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0xb800
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0xffffb800, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0xb8,0xff,0xff]
define <2 x i16> @mul_inline_imm_neg_0.5_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half -0.5, half -0.5> to <2 x i16>)
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}mul_inline_imm_1.0_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0x3c00
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0x3c00, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0x3c,0x00,0x00]
define <2 x i16> @mul_inline_imm_1.0_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half 1.0, half 1.0> to <2 x i16>)
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}mul_inline_imm_neg_1.0_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0xbc00
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0xffffbc00, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0xbc,0xff,0xff]
define <2 x i16> @mul_inline_imm_neg_1.0_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half -1.0, half -1.0> to <2 x i16>)
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}shl_inline_imm_2.0_v2i16:
; GFX9: v_pk_lshlrev_b16 v0, v0, 2.0 op_sel:[0,1]

; GFX10: v_pk_lshlrev_b16 v0, v0, 2.0 op_sel:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xe9,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}}]
define <2 x i16> @shl_inline_imm_2.0_v2i16(<2 x i16> %x) {
  %y = shl <2 x i16> bitcast (<2 x half> <half 2.0, half 2.0> to <2 x i16>), %x
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}shl_inline_imm_neg_2.0_v2i16:
; GFX9: v_pk_lshlrev_b16 v0, v0, -2.0 op_sel:[0,1]

; GFX10: v_pk_lshlrev_b16 v0, v0, -2.0 op_sel:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xeb,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}}]
define <2 x i16> @shl_inline_imm_neg_2.0_v2i16(<2 x i16> %x) {
  %y = shl <2 x i16> bitcast (<2 x half> <half -2.0, half -2.0> to <2 x i16>), %x
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}mul_inline_imm_4.0_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0x4400
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0x4400, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0x44,0x00,0x00]
define <2 x i16> @mul_inline_imm_4.0_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half 4.0, half 4.0> to <2 x i16>)
  ret <2 x i16> %y

}

; GCN-LABEL: {{^}}mul_inline_imm_neg_4.0_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0xc400
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0xffffc400, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x00,0xc4,0xff,0xff]
define <2 x i16> @mul_inline_imm_neg_4.0_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half -4.0, half -4.0> to <2 x i16>)
  ret <2 x i16> %y
}

; GCN-LABEL: {{^}}mul_inline_imm_inv2pi_v2i16:
; GFX9: s_movk_i32 [[K:s[0-9]+]], 0x3118
; GFX9: v_pk_mul_lo_u16 v0, v0, [[K]] op_sel_hi:[1,0]

; GFX10: v_pk_mul_lo_u16 v0, 0x3118, v0 op_sel_hi:[0,1] ; encoding: [0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0xff,0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x{{[0-9a-f]+}},0x18,0x31,0x00,0x00]
define <2 x i16> @mul_inline_imm_inv2pi_v2i16(<2 x i16> %x) {
  %y = mul <2 x i16> %x, bitcast (<2 x half> <half 0xH3118, half 0xH3118> to <2 x i16>)
  ret <2 x i16> %y
}

attributes #0 = { nounwind }

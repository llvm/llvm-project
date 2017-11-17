; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=gfx901 -mattr=-flat-for-global -verify-machineinstrs -enable-packed-inlinable-literals < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn--amdhsa -mcpu=kaveri -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=CI %s
; FIXME: Merge into imm.ll

; GCN-LABEL: {{^}}store_inline_imm_neg_0.0_v2i16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_neg_0.0_v2i16(<2 x i16> addrspace(1)* %out) #0 {
  store <2 x i16> <i16 -32768, i16 -32768>, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_0.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 0.0, half 0.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_imm_neg_0.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x80008000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_imm_neg_0.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half -0.0, half -0.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_0.5_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x38003800{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_0.5_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 0.5, half 0.5>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_0.5_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb800b800{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_0.5_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half -0.5, half -0.5>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_1.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x3c003c00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_1.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 1.0, half 1.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_1.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xbc00bc00{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_1.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half -1.0, half -1.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_2.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x40004000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_2.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 2.0, half 2.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_2.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc000c000{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_2.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half -2.0, half -2.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_4.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x44004400{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_4.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 4.0, half 4.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_4.0_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xc400c400{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_4.0_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half -4.0, half -4.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_inv_2pi_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x31183118{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_inv_2pi_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 0xH3118, half 0xH3118>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_inline_imm_m_inv_2pi_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0xb118b118{{$}}
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_inline_imm_m_inv_2pi_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 0xHB118, half 0xHB118>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}store_literal_imm_v2f16:
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 0x6c006c00
; GCN: buffer_store_dword [[REG]]
define amdgpu_kernel void @store_literal_imm_v2f16(<2 x half> addrspace(1)* %out) #0 {
  store <2 x half> <half 4096.0, half 4096.0>, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST0:v[0-9]+]], 0
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST0]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_0.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0.0, half 0.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_0.5_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 0.5, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST05:v[0-9]+]], 0x3800
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_0.5_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0.5, half 0.5>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_0.5_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -0.5{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -0.5, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM05:v[0-9]+]], 0xb800
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_0.5_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -0.5, half -0.5>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 1.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST1:v[0-9]+]], 0x3c00
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_1.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 1.0, half 1.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -1.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -1.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM1:v[0-9]+]], 0xbc00
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_1.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -1.0, half -1.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 2.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 0x4000
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_2.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 2.0, half 2.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -2.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -2.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM2:v[0-9]+]], 0xc000
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_2.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -2.0, half -2.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_4.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 4.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 4.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST4:v[0-9]+]], 0x4400
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST4]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_4.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 4.0, half 4.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_4.0_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -4.0{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -4.0, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM4:v[0-9]+]], 0xc400
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM4]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_4.0_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half -4.0, half -4.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_inline_imm_0.5_v2f16:
; GFX9: buffer_load_dword [[VAL:v[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 0.5
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_dword
; VI-NOT: and
; VI: v_mov_b32_e32 [[CONST05:v[0-9]+]], 0x3800
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[CONST05]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 0.5, v{{[0-9]+}}
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @commute_add_inline_imm_0.5_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %x = load <2 x half>, <2 x half> addrspace(1)* %in
  %y = fadd <2 x half> %x, <half 0.5, half 0.5>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}commute_add_literal_v2f16:
; GFX9-DAG: buffer_load_dword [[VAL:v[0-9]+]]
; GFX9-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x6400{{$}}
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], [[K]] op_sel_hi:[1,0]{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x6400{{$}}
; VI-DAG: buffer_load_dword
; VI-NOT: and
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, [[K]]
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, [[K]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: buffer_store_dword
define amdgpu_kernel void @commute_add_literal_v2f16(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %x = load <2 x half>, <2 x half> addrspace(1)* %in
  %y = fadd <2 x half> %x, <half 1024.0, half 1024.0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_1_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 1{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 1, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST1:v[0-9]+]], 1
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_1_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0001, half 0xH0001>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_2_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 2{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 2, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST2:v[0-9]+]], 2
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_2_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0002, half 0xH0002>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_16_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 16{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 16, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST16:v[0-9]+]], 16
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_16_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0010, half 0xH0010>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_1_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -1{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -1, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM1:v[0-9]+]], 0xffff
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_1_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xHFFFF, half 0xHFFFF>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_2_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -2{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -2, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM2:v[0-9]+]], 0xfffe
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_2_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xHFFFE, half 0xHFFFE>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_neg_16_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], -16{{$}}
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, -16, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONSTM16:v[0-9]+]], 0xfff0
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONSTM16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_neg_16_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xHFFF0, half 0xHFFF0>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_63_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 63
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 63, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST63:v[0-9]+]], 63
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST63]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_63_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH003F, half 0xH003F>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}add_inline_imm_64_v2f16:
; GFX9: s_load_dword [[VAL:s[0-9]+]]
; GFX9: v_pk_add_f16 [[REG:v[0-9]+]], [[VAL]], 64
; GFX9: buffer_store_dword [[REG]]

; VI: buffer_load_ushort [[VAL0:v[0-9]+]]
; VI: buffer_load_ushort [[VAL1:v[0-9]+]]
; VI-DAG: v_add_f16_e32 v{{[0-9]+}}, 64, [[VAL0]]
; VI-DAG: v_mov_b32_e32 [[CONST64:v[0-9]+]], 64
; VI-DAG: v_add_f16_sdwa v{{[0-9]+}}, [[VAL1]], [[CONST64]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
; VI: v_or_b32
; VI: buffer_store_dword
define amdgpu_kernel void @add_inline_imm_64_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %x) #0 {
  %y = fadd <2 x half> %x, <half 0xH0040, half 0xH0040>
  store <2 x half> %y, <2 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }

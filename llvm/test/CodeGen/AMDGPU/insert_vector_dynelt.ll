; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s

; GCN-LABEL: {{^}}float4_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_LAST:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC4:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_FIRST:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC4]]
; GCN:     flat_store_dwordx4 v[{{[0-9:]+}}], v{{\[}}[[ELT_FIRST]]:[[ELT_LAST]]]
define amdgpu_kernel void @float4_inselt(<4 x float> addrspace(1)* %out, <4 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <4 x float> %vec, float 1.000000e+00, i32 %sel
  store <4 x float> %v, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float4_inselt_undef:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-NOT: v_cmp_
; GCN-NOT: v_cndmask_
; GCN:     v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
; GCN:     v_mov_b32_e32 v{{[0-9]+}}, [[ONE]]
define amdgpu_kernel void @float4_inselt_undef(<4 x float> addrspace(1)* %out, i32 %sel) {
entry:
  %v = insertelement <4 x float> undef, float 1.000000e+00, i32 %sel
  store <4 x float> %v, <4 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}int4_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_LAST:[0-9]+]], 1, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC4:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_FIRST:[0-9]+]], 1, v{{[0-9]+}}, [[CC4]]
; GCN:     flat_store_dwordx4 v[{{[0-9:]+}}], v{{\[}}[[ELT_FIRST]]:[[ELT_LAST]]]
define amdgpu_kernel void @int4_inselt(<4 x i32> addrspace(1)* %out, <4 x i32> %vec, i32 %sel) {
entry:
  %v = insertelement <4 x i32> %vec, i32 1, i32 %sel
  store <4 x i32> %v, <4 x i32> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float2_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_LAST:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_FIRST:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC2]]
; GCN:     flat_store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[ELT_FIRST]]:[[ELT_LAST]]]
define amdgpu_kernel void @float2_inselt(<2 x float> addrspace(1)* %out, <2 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <2 x float> %vec, float 1.000000e+00, i32 %sel
  store <2 x float> %v, <2 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float8_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], [[IDX:s[0-9]+]], 3
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_LAST0:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC3:[^,]+]], [[IDX]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC3]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC4:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_FIRST0:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC4]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC5:[^,]+]], [[IDX:s[0-9]+]], 7
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_LAST1:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC5]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC6:[^,]+]], [[IDX]], 6
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC6]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC7:[^,]+]], [[IDX]], 5
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1.0, v{{[0-9]+}}, [[CC7]]
; GCN-DAG: v_cmp_ne_u32_e64 [[CC8:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cndmask_b32_e32 v[[ELT_FIRST1:[0-9]+]], 1.0, v{{[0-9]+}}, [[CC8]]
; GCN-DAG: flat_store_dwordx4 v[{{[0-9:]+}}], v{{\[}}[[ELT_FIRST0]]:[[ELT_LAST0]]]
; GCN-DAG: flat_store_dwordx4 v[{{[0-9:]+}}], v{{\[}}[[ELT_FIRST1]]:[[ELT_LAST1]]]
define amdgpu_kernel void @float8_inselt(<8 x float> addrspace(1)* %out, <8 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <8 x float> %vec, float 1.000000e+00, i32 %sel
  store <8 x float> %v, <8 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float16_inselt:
; GCN: v_movreld_b32
define amdgpu_kernel void @float16_inselt(<16 x float> addrspace(1)* %out, <16 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <16 x float> %vec, float 1.000000e+00, i32 %sel
  store <16 x float> %v, <16 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float32_inselt:
; GCN: v_movreld_b32
define amdgpu_kernel void @float32_inselt(<32 x float> addrspace(1)* %out, <32 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <32 x float> %vec, float 1.000000e+00, i32 %sel
  store <32 x float> %v, <32 x float> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}half4_inselt:
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN:     s_lshl_b32 [[SEL:s[0-9]+]], s{{[0-9]+}}, 4
; GCN:     s_lshl_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], [[SEL]]
; GCN:     s_mov_b32 [[K:s[0-9]+]], 0x3c003c00
; GCN:     v_mov_b32_e32 [[V:v[0-9]+]], [[K]]
; GCN:     v_bfi_b32 v{{[0-9]+}}, s{{[0-9]+}}, [[V]], v{{[0-9]+}}
; GCN:     v_bfi_b32 v{{[0-9]+}}, s{{[0-9]+}}, [[V]], v{{[0-9]+}}
define amdgpu_kernel void @half4_inselt(<4 x half> addrspace(1)* %out, <4 x half> %vec, i32 %sel) {
entry:
  %v = insertelement <4 x half> %vec, half 1.000000e+00, i32 %sel
  store <4 x half> %v, <4 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}half2_inselt:
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN:     s_lshl_b32 [[SEL:s[0-9]+]], s{{[0-9]+}}, 4
; GCN:     s_lshl_b32 [[V:s[0-9]+]], 0xffff, [[SEL]]
; GCN:     v_bfi_b32 v{{[0-9]+}}, [[V]], v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @half2_inselt(<2 x half> addrspace(1)* %out, <2 x half> %vec, i32 %sel) {
entry:
  %v = insertelement <2 x half> %vec, half 1.000000e+00, i32 %sel
  store <2 x half> %v, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}half8_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 0
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 1
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 2
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 3
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 4
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 5
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 6
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 7
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
define amdgpu_kernel void @half8_inselt(<8 x half> addrspace(1)* %out, <8 x half> %vec, i32 %sel) {
entry:
  %v = insertelement <8 x half> %vec, half 1.000000e+00, i32 %sel
  store <8 x half> %v, <8 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}short2_inselt:
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN:     v_mov_b32_e32 [[K:v[0-9]+]], 0x10001
; GCN:     s_lshl_b32 [[SEL:s[0-9]+]], s{{[0-9]+}}, 4
; GCN:     s_lshl_b32 [[V:s[0-9]+]], 0xffff, [[SEL]]
; GCN:     v_bfi_b32 v{{[0-9]+}}, [[V]], [[K]], v{{[0-9]+}}
define amdgpu_kernel void @short2_inselt(<2 x i16> addrspace(1)* %out, <2 x i16> %vec, i32 %sel) {
entry:
  %v = insertelement <2 x i16> %vec, i16 1, i32 %sel
  store <2 x i16> %v, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}short4_inselt:
; GCN-NOT: v_cndmask_b32
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN:     s_lshl_b32 [[SEL:s[0-9]+]], s{{[0-9]+}}, 4
; GCN:     s_lshl_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], [[SEL]]
; GCN:     s_mov_b32 [[K:s[0-9]+]], 0x10001
; GCN:     v_mov_b32_e32 [[V:v[0-9]+]], [[K]]
; GCN:     v_bfi_b32 v{{[0-9]+}}, s{{[0-9]+}}, [[V]], v{{[0-9]+}}
; GCN:     v_bfi_b32 v{{[0-9]+}}, s{{[0-9]+}}, [[V]], v{{[0-9]+}}
define amdgpu_kernel void @short4_inselt(<4 x i16> addrspace(1)* %out, <4 x i16> %vec, i32 %sel) {
entry:
  %v = insertelement <4 x i16> %vec, i16 1, i32 %sel
  store <4 x i16> %v, <4 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}byte8_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN:     s_lshl_b32 [[SEL:s[0-9]+]], s{{[0-9]+}}, 3
; GCN:     s_lshl_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], [[SEL]]
; GCN:     s_mov_b32 [[K:s[0-9]+]], 0x1010101
; GCN:     s_and_b32 s3, s1, [[K]]
; GCN:     s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, [[K]]
; GCN:     s_andn2_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GCN:     s_or_b64 s[{{[0-9:]+}}], s[{{[0-9:]+}}], s[{{[0-9:]+}}]
define amdgpu_kernel void @byte8_inselt(<8 x i8> addrspace(1)* %out, <8 x i8> %vec, i32 %sel) {
entry:
  %v = insertelement <8 x i8> %vec, i8 1, i32 %sel
  store <8 x i8> %v, <8 x i8> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}byte16_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 0
; GCN-DAG: v_cmp_ne_u32_e64 {{[^,]+}}, {{s[0-9]+}}, 15
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_cndmask_b32_e32
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
; GCN-DAG: v_or_b32_sdwa
define amdgpu_kernel void @byte16_inselt(<16 x i8> addrspace(1)* %out, <16 x i8> %vec, i32 %sel) {
entry:
  %v = insertelement <16 x i8> %vec, i8 1, i32 %sel
  store <16 x i8> %v, <16 x i8> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double2_inselt:
; GCN-NOT: v_movrel
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[CC1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC1]]
; GCN-DAG: v_cmp_eq_u32_e64 [[CC2:[^,]+]], [[IDX]], 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[CC2]]
; GCN-DAG: v_cndmask_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 0, [[CC2]]
define amdgpu_kernel void @double2_inselt(<2 x double> addrspace(1)* %out, <2 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <2 x double> %vec, double 1.000000e+00, i32 %sel
  store <2 x double> %v, <2 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double8_inselt:
; GCN-NOT: v_cndmask
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movreld_b32_e32 v[[#BASE:]], 0
; GCN-NOT: s_mov_b32 m0
; GCN:     v_movreld_b32_e32 v[[#BASE+1]],
define amdgpu_kernel void @double8_inselt(<8 x double> addrspace(1)* %out, <8 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <8 x double> %vec, double 1.000000e+00, i32 %sel
  store <8 x double> %v, <8 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double7_inselt:
; GCN-NOT: v_cndmask
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movreld_b32_e32 v[[#BASE]], 0
; GCN-NOT: s_mov_b32 m0
; GCN:     v_movreld_b32_e32 v[[#BASE+1]],
define amdgpu_kernel void @double7_inselt(<7 x double> addrspace(1)* %out, <7 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <7 x double> %vec, double 1.000000e+00, i32 %sel
  store <7 x double> %v, <7 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double16_inselt:
; GCN-NOT: v_cndmask
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movreld_b32_e32 v[[#BASE:]], 0
; GCN-NOT: s_mov_b32 m0
; GCN:     v_movreld_b32_e32 v[[#BASE+1]],
define amdgpu_kernel void @double16_inselt(<16 x double> addrspace(1)* %out, <16 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <16 x double> %vec, double 1.000000e+00, i32 %sel
  store <16 x double> %v, <16 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double15_inselt:
; GCN-NOT: v_cndmask
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movreld_b32_e32 v[[#BASE:]], 0
; GCN-NOT: s_mov_b32 m0
; GCN:     v_movreld_b32_e32 v[[#BASE+1]],
define amdgpu_kernel void @double15_inselt(<15 x double> addrspace(1)* %out, <15 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <15 x double> %vec, double 1.000000e+00, i32 %sel
  store <15 x double> %v, <15 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}bit4_inselt:
; GCN: buffer_store_byte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
define amdgpu_kernel void @bit4_inselt(<4 x i1> addrspace(1)* %out, <4 x i1> %vec, i32 %sel) {
entry:
  %v = insertelement <4 x i1> %vec, i1 1, i32 %sel
  store <4 x i1> %v, <4 x i1> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}bit128_inselt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_ne_u32_e64 [[CC1:[^,]+]], s{{[0-9]+}}, 0
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}, [[CC1]]
; GCN-DAG: v_mov_b32_e32 [[LASTIDX:v[0-9]+]], 0x7f
; GCN-DAG: v_cmp_ne_u32_e32 [[CCL:[^,]+]], s{{[0-9]+}}, [[LASTIDX]]
; GCN-DAG: v_cndmask_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}, [[CCL]]
define amdgpu_kernel void @bit128_inselt(<128 x i1> addrspace(1)* %out, <128 x i1> %vec, i32 %sel) {
entry:
  %v = insertelement <128 x i1> %vec, i1 1, i32 %sel
  store <128 x i1> %v, <128 x i1> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float32_inselt_vec:
; GCN-NOT: buffer_
; GCN-COUNT-32: v_cmp_ne_u32
; GCN-COUNT-32: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, 1.0,
define amdgpu_ps <32 x float> @float32_inselt_vec(<32 x float> %vec, i32 %sel) {
entry:
  %v = insertelement <32 x float> %vec, float 1.000000e+00, i32 %sel
  ret <32 x float> %v
}

; GCN-LABEL: {{^}}double8_inselt_vec:
; GCN-NOT: buffer_
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
; GCN:         v_cmp_eq_u32
; GCN-COUNT-2: v_cndmask_b32
define <8 x double> @double8_inselt_vec(<8 x double> %vec, i32 %sel) {
entry:
  %v = insertelement <8 x double> %vec, double 1.000000e+00, i32 %sel
  ret <8 x double> %v
}

; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s

; GCN-LABEL: {{^}}float4_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_ne_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_ne_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], 0, 1.0, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], 2.0, [[V1]], [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], 4.0, [[V2]], [[C3]]
; GCN:     store_dword v[{{[0-9:]+}}], [[V3]]
define amdgpu_kernel void @float4_extelt(float addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x float> <float 0.0, float 1.0, float 2.0, float 4.0>, i32 %sel
  store float %ext, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}int4_extelt:
; GCN-NOT: buffer_
; GCN-DAG: s_cmp_lg_u32 [[IDX:s[0-9]+]], 2
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX]], 1
; GCN-DAG: s_cmp_lg_u32 [[IDX]], 3
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], 0, 1, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], 2, [[V1]], vcc
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], 4, [[V2]], vcc
; GCN: store_dword v[{{[0-9:]+}}], [[V3]]
define amdgpu_kernel void @int4_extelt(i32 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x i32> <i32 0, i32 1, i32 2, i32 4>, i32 %sel
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double4_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_eq_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_eq_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C3]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @double4_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x double> <double 0.01, double 1.01, double 2.01, double 4.01>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double5_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_eq_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_eq_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cmp_eq_u32_e64 [[C4:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C4]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @double5_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <5 x double> <double 0.01, double 1.01, double 2.01, double 4.01, double 5.01>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}half4_extelt:
; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x40003c00
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x44004200
; GCN-DAG: s_lshl_b32 [[SEL:s[0-p]+]], s{{[0-9]+}}, 4
; GCN:     s_lshr_b64 s{{\[}}[[RL:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SL]]:[[SH]]], [[SEL]]
; GCN-DAG: v_mov_b32_e32 v[[VRL:[0-9]+]], s[[RL]]
; GCN:     store_short v[{{[0-9:]+}}], v[[VRL]]
define amdgpu_kernel void @half4_extelt(half addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x half> <half 1.0, half 2.0, half 3.0, half 4.0>, i32 %sel
  store half %ext, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float2_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], 0, 1.0, [[C1]]
; GCN: store_dword v[{{[0-9:]+}}], [[V1]]
define amdgpu_kernel void @float2_extelt(float addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <2 x float> <float 0.0, float 1.0>, i32 %sel
  store float %ext, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double2_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @double2_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <2 x double> <double 0.01, double 1.01>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}half8_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_ne_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_ne_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cmp_ne_u32_e64 [[C4:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cmp_ne_u32_e64 [[C5:[^,]+]], [[IDX]], 5
; GCN-DAG: v_cmp_ne_u32_e64 [[C6:[^,]+]], [[IDX]], 6
; GCN-DAG: v_cmp_ne_u32_e64 [[C7:[^,]+]], [[IDX]], 7
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], {{[^,]+}}, [[V1]], [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], {{[^,]+}}, [[V2]], [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V4:v[0-9]+]], {{[^,]+}}, [[V3]], [[C4]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V5:v[0-9]+]], {{[^,]+}}, [[V4]], [[C5]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V6:v[0-9]+]], {{[^,]+}}, [[V5]], [[C6]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V7:v[0-9]+]], {{[^,]+}}, [[V6]], [[C7]]
; GCN:     store_short v[{{[0-9:]+}}], [[V7]]
define amdgpu_kernel void @half8_extelt(half addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <8 x half> <half 1.0, half 2.0, half 3.0, half 4.0, half 5.0, half 6.0, half 7.0, half 8.0>, i32 %sel
  store half %ext, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}short8_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_ne_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_ne_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cmp_ne_u32_e64 [[C4:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cmp_ne_u32_e64 [[C5:[^,]+]], [[IDX]], 5
; GCN-DAG: v_cmp_ne_u32_e64 [[C6:[^,]+]], [[IDX]], 6
; GCN-DAG: v_cmp_ne_u32_e64 [[C7:[^,]+]], [[IDX]], 7
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], {{[^,]+}}, [[V1]], [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], {{[^,]+}}, [[V2]], [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V4:v[0-9]+]], {{[^,]+}}, [[V3]], [[C4]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V5:v[0-9]+]], {{[^,]+}}, [[V4]], [[C5]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V6:v[0-9]+]], {{[^,]+}}, [[V5]], [[C6]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V7:v[0-9]+]], {{[^,]+}}, [[V6]], [[C7]]
; GCN:     store_short v[{{[0-9:]+}}], [[V7]]
define amdgpu_kernel void @short8_extelt(i16 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>, i32 %sel
  store i16 %ext, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float8_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_ne_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_ne_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cmp_ne_u32_e64 [[C4:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cmp_ne_u32_e64 [[C5:[^,]+]], [[IDX]], 5
; GCN-DAG: v_cmp_ne_u32_e64 [[C6:[^,]+]], [[IDX]], 6
; GCN-DAG: v_cmp_ne_u32_e64 [[C7:[^,]+]], [[IDX]], 7
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], {{[^,]+}}, [[V1]], [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], {{[^,]+}}, [[V2]], [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V4:v[0-9]+]], {{[^,]+}}, [[V3]], [[C4]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V5:v[0-9]+]], {{[^,]+}}, [[V4]], [[C5]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V6:v[0-9]+]], {{[^,]+}}, [[V5]], [[C6]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V7:v[0-9]+]], {{[^,]+}}, [[V6]], [[C7]]
; GCN:     store_dword v[{{[0-9:]+}}], [[V7]]
define amdgpu_kernel void @float8_extelt(float addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <8 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0>, i32 %sel
  store float %ext, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double8_extelt:
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v[[#BASE:]], [[ZERO]]
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_LO:[0-9]+]], v[[#BASE]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_HI:[0-9]+]], v[[#BASE+1]]
; GCN:     store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[RES_LO]]:[[RES_HI]]]
define amdgpu_kernel void @double8_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <8 x double> <double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double7_extelt:
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v[[#BASE:]], [[ZERO]]
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_LO:[0-9]+]], v[[#BASE]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_HI:[0-9]+]], v[[#BASE+1]]
; GCN:     store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[RES_LO]]:[[RES_HI]]]
define amdgpu_kernel void @double7_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <7 x double> <double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float16_extelt:
; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 m0,
; GCN-DAG: v_mov_b32_e32 [[VLO:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40a00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40c00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40e00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41000000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41100000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41200000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41300000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41500000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41600000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41700000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41800000
; GCN-DAG: v_movrels_b32_e32 [[RES:v[0-9]+]], [[VLO]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]
define amdgpu_kernel void @float16_extelt(float addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <16 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0>, i32 %sel
  store float %ext, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double15_extelt:
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v[[#BASE:]], [[ZERO]]
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_LO:[0-9]+]], v[[#BASE]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_HI:[0-9]+]], v[[#BASE+1]]
; GCN:     store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[RES_LO]]:[[RES_HI]]]
define amdgpu_kernel void @double15_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <15 x double> <double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0, double 9.0, double 10.0, double 11.0, double 12.0, double 13.0, double 14.0, double 15.0>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}double16_extelt:
; GCN-NOT: buffer_
; GCN-NOT: s_or_b32
; GCN-DAG: s_mov_b32 [[ZERO:s[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 v[[#BASE:]], [[ZERO]]
; GCN-DAG: s_mov_b32 m0, [[IND:s[0-9]+]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_LO:[0-9]+]], v[[#BASE]]
; GCN-DAG: v_movrels_b32_e32 v[[RES_HI:[0-9]+]], v[[#BASE+1]]
; GCN:     store_dwordx2 v[{{[0-9:]+}}], v{{\[}}[[RES_LO]]:[[RES_HI]]]
define amdgpu_kernel void @double16_extelt(double addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <16 x double> <double 1.0, double 2.0, double 3.0, double 4.0, double 5.0, double 6.0, double 7.0, double 8.0, double 9.0, double 10.0, double 11.0, double 12.0, double 13.0, double 14.0, double 15.0, double 16.0>, i32 %sel
  store double %ext, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float32_extelt:
; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 m0,
; GCN-DAG: v_mov_b32_e32 [[VLO:v[0-9]+]], 1.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 2.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 4.0
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40a00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40c00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x40e00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41000000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41100000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41200000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41300000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41400000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41500000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41600000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41700000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41800000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41880000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41900000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41980000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41a00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41a80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41b00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41b80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41c00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41c80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41d00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41d80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41e00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41e80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41f00000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x41f80000
; GCN-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x42000000
; GCN-DAG: v_movrels_b32_e32 [[RES:v[0-9]+]], [[VLO]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]
define amdgpu_kernel void @float32_extelt(float addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <32 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0, float 17.0, float 18.0, float 19.0, float 20.0, float 21.0, float 22.0, float 23.0, float 24.0, float 25.0, float 26.0, float 27.0, float 28.0, float 29.0, float 30.0, float 31.0, float 32.0>, i32 %sel
  store float %ext, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}byte8_extelt:
; GCN-NOT: buffer_
; GCN-DAG: s_mov_b32 s[[SL:[0-9]+]], 0x4030201
; GCN-DAG: s_mov_b32 s[[SH:[0-9]+]], 0x8070605
; GCN-DAG: s_lshl_b32 [[SEL:s[0-p]+]], s{{[0-9]+}}, 3
; GCN:     s_lshr_b64 s{{\[}}[[RL:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SL]]:[[SH]]], [[SEL]]
; GCN-DAG: v_mov_b32_e32 v[[VRL:[0-9]+]], s[[RL]]
; GCN:     store_byte v[{{[0-9:]+}}], v[[VRL]]
define amdgpu_kernel void @byte8_extelt(i8 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <8 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, i32 %sel
  store i8 %ext, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}byte16_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e64 [[C1:[^,]+]], [[IDX:s[0-9]+]], 1
; GCN-DAG: v_cmp_ne_u32_e64 [[C2:[^,]+]], [[IDX]], 2
; GCN-DAG: v_cmp_ne_u32_e64 [[C3:[^,]+]], [[IDX]], 3
; GCN-DAG: v_cmp_ne_u32_e64 [[C4:[^,]+]], [[IDX]], 4
; GCN-DAG: v_cmp_ne_u32_e64 [[C5:[^,]+]], [[IDX]], 5
; GCN-DAG: v_cmp_ne_u32_e64 [[C6:[^,]+]], [[IDX]], 6
; GCN-DAG: v_cmp_ne_u32_e64 [[C7:[^,]+]], [[IDX]], 7
; GCN-DAG: v_cmp_ne_u32_e64 [[C8:[^,]+]], [[IDX]], 8
; GCN-DAG: v_cmp_ne_u32_e64 [[C9:[^,]+]], [[IDX]], 9
; GCN-DAG: v_cmp_ne_u32_e64 [[C10:[^,]+]], [[IDX]], 10
; GCN-DAG: v_cmp_ne_u32_e64 [[C11:[^,]+]], [[IDX]], 11
; GCN-DAG: v_cmp_ne_u32_e64 [[C12:[^,]+]], [[IDX]], 12
; GCN-DAG: v_cmp_ne_u32_e64 [[C13:[^,]+]], [[IDX]], 13
; GCN-DAG: v_cmp_ne_u32_e64 [[C14:[^,]+]], [[IDX]], 14
; GCN-DAG: v_cmp_ne_u32_e64 [[C15:[^,]+]], [[IDX]], 15
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], {{[^,]+}}, {{[^,]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V2:v[0-9]+]], {{[^,]+}}, [[V1]], [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V3:v[0-9]+]], {{[^,]+}}, [[V2]], [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V4:v[0-9]+]], {{[^,]+}}, [[V3]], [[C4]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V5:v[0-9]+]], {{[^,]+}}, [[V4]], [[C5]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V6:v[0-9]+]], {{[^,]+}}, [[V5]], [[C6]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V7:v[0-9]+]], {{[^,]+}}, [[V6]], [[C7]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V8:v[0-9]+]], {{[^,]+}}, [[V7]], [[C8]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V9:v[0-9]+]], {{[^,]+}}, [[V8]], [[C8]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V10:v[0-9]+]], {{[^,]+}}, [[V9]], [[C10]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V11:v[0-9]+]], {{[^,]+}}, [[V10]], [[C11]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V12:v[0-9]+]], {{[^,]+}}, [[V11]], [[C12]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V13:v[0-9]+]], {{[^,]+}}, [[V12]], [[C13]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V14:v[0-9]+]], {{[^,]+}}, [[V13]], [[C14]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V15:v[0-9]+]], {{[^,]+}}, [[V14]], [[C15]]
; GCN:     store_byte v[{{[0-9:]+}}], [[V15]]
define amdgpu_kernel void @byte16_extelt(i8 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, i32 %sel
  store i8 %ext, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}bit4_extelt:
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG: buffer_store_byte [[ZERO]],
; GCN-DAG: buffer_store_byte [[ONE]],
; GCN-DAG: buffer_store_byte [[ZERO]],
; GCN-DAG: buffer_store_byte [[ONE]],
; GCN:     buffer_load_ubyte [[LOAD:v[0-9]+]],
; GCN:     v_and_b32_e32 [[RES:v[0-9]+]], 1, [[LOAD]]
; GCN:     flat_store_dword v[{{[0-9:]+}}], [[RES]]
define amdgpu_kernel void @bit4_extelt(i32 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <4 x i1> <i1 0, i1 1, i1 0, i1 1>, i32 %sel
  %zext = zext i1 %ext to i32
  store i32 %zext, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}bit128_extelt:
; GCN-NOT: buffer_
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], 0, 1
; GCN-DAG: v_mov_b32_e32 [[LASTIDX:v[0-9]+]], 0x7f
; GCN-DAG: v_cmp_ne_u32_e32 [[CL:[^,]+]], s{{[0-9]+}}, [[LASTIDX]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[VL:v[0-9]+]], 0, [[V1]], [[CL]]
; GCN:     v_and_b32_e32 [[RES:v[0-9]+]], 1, [[VL]]
; GCN:     store_dword v[{{[0-9:]+}}], [[RES]]
define amdgpu_kernel void @bit128_extelt(i32 addrspace(1)* %out, i32 %sel) {
entry:
  %ext = extractelement <128 x i1> <i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0, i1 1, i1 0>, i32 %sel
  %zext = zext i1 %ext to i32
  store i32 %zext, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}float32_extelt_vec:
; GCN-NOT: buffer_
; GCN-DAG: v_cmp_eq_u32_e{{32|64}} [[CC1:[^,]+]], 1, v0
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[V1:v[0-9]+]], 1.0, 2.0, [[CC1]]
; GCN-DAG: v_mov_b32_e32 [[LASTVAL:v[0-9]+]], 0x42000000
; GCN-DAG: v_cmp_ne_u32_e32 [[LASTCC:[^,]+]], 31, v0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v0, [[LASTVAL]], v{{[0-9]+}}, [[LASTCC]]
define float @float32_extelt_vec(i32 %sel) {
entry:
  %ext = extractelement <32 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0, float 17.0, float 18.0, float 19.0, float 20.0, float 21.0, float 22.0, float 23.0, float 24.0, float 25.0, float 26.0, float 27.0, float 28.0, float 29.0, float 30.0, float 31.0, float 32.0>, i32 %sel
  ret float %ext
}

; GCN-LABEL: {{^}}double16_extelt_vec:
; GCN-NOT: buffer_
; GCN-DAG: v_mov_b32_e32 [[V1HI:v[0-9]+]], 0x3ff19999
; GCN-DAG: v_mov_b32_e32 [[V1LO:v[0-9]+]], 0x9999999a
; GCN-DAG: v_mov_b32_e32 [[V2HI:v[0-9]+]], 0x4000cccc
; GCN-DAG: v_mov_b32_e32 [[V2LO:v[0-9]+]], 0xcccccccd
; GCN-DAG: v_cmp_eq_u32_e{{32|64}} [[CC1:[^,]+]], 1, v0
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[R1HI:v[0-9]+]], [[V1HI]], [[V2HI]], [[CC1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} [[R1LO:v[0-9]+]], [[V1LO]], [[V2LO]], [[CC1]]
define double @double16_extelt_vec(i32 %sel) {
entry:
  %ext = extractelement <16 x double> <double 1.1, double 2.1, double 3.1, double 4.1, double 5.1, double 6.1, double 7.1, double 8.1, double 9.1, double 10.1, double 11.1, double 12.1, double 13.1, double 14.1, double 15.1, double 16.1>, i32 %sel
  ret double %ext
}

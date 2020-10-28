; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=GCN,UNPACKED,PREGFX10,PREGFX10-UNPACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PREGFX10,PREGFX10-PACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,PREGFX10,PREGFX10-PACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,GFX10,GFX10-PACKED %s


; GCN-LABEL: {{^}}tbuffer_store_d16_x:
; GCN-DAG: s_load_dwordx4
; GCN-DAG: s_load_dword s[[S_LO:[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[S_LO]]
; PREGFX10: tbuffer_store_format_d16_x v[[V_LO]], off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_NUM_FORMAT_USCALED]
; GFX10: tbuffer_store_format_d16_x v[[V_LO]], off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_FMT_10_11_11_SSCALED]
define amdgpu_kernel void @tbuffer_store_d16_x(<4 x i32> %rsrc, half %data) {
main_body:
  call void @llvm.amdgcn.raw.tbuffer.store.f16(half %data, <4 x i32> %rsrc, i32 0, i32 0, i32 33, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_d16_xy:
; GCN: s_load_dword [[S_DATA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}},
; UNPACKED-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[S_DATA]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED:s[0-9]+]], [[S_DATA]], 0xffff{{$}}
; UNPACKED-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[MASKED]]
; UNPACKED-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], [[SHR]]
; PREGFX10-UNPACKED: tbuffer_store_format_d16_xy v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_NUM_FORMAT_USCALED]

; PREGFX10-PACKED: tbuffer_store_format_d16_xy v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_NUM_FORMAT_USCALED]
; GFX10-PACKED: tbuffer_store_format_d16_xy v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_FMT_10_11_11_SSCALED]
define amdgpu_kernel void @tbuffer_store_d16_xy(<4 x i32> %rsrc, <2 x half> %data) {
main_body:
  call void @llvm.amdgcn.raw.tbuffer.store.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 0, i32 0, i32 33, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_d16_xyzw:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[S_DATA_0:[0-9]+]]:[[S_DATA_1:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}},

; UNPACKED-DAG: s_mov_b32 [[K:s[0-9]+]], 0xffff{{$}}
; UNPACKED-DAG: s_lshr_b32 [[SHR0:s[0-9]+]], s[[S_DATA_0]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED0:s[0-9]+]], s[[S_DATA_0]], [[K]]
; UNPACKED-DAG: s_lshr_b32 [[SHR1:s[0-9]+]], s[[S_DATA_1]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED1:s[0-9]+]], s[[S_DATA_1]], [[K]]

; UNPACKED-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], [[MASKED0]]
; UNPACKED-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[SHR1]]
; PREGFX10-UNPACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_NUM_FORMAT_USCALED]


; PACKED-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s[[S_DATA_0]]
; PACKED-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s[[S_DATA_1]]
; PREGFX10-PACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_NUM_FORMAT_USCALED]
; GFX10-PACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0 format:[BUF_FMT_10_11_11_SSCALED]
define amdgpu_kernel void @tbuffer_store_d16_xyzw(<4 x i32> %rsrc, <4 x half> %data) {
main_body:
  call void @llvm.amdgcn.raw.tbuffer.store.v4f16(<4 x half> %data, <4 x i32> %rsrc, i32 0, i32 0, i32 33, i32 0)
  ret void
}

declare void @llvm.amdgcn.raw.tbuffer.store.f16(half, <4 x i32>, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.tbuffer.store.v2f16(<2 x half>, <4 x i32>, i32, i32, i32, i32)
declare void @llvm.amdgcn.raw.tbuffer.store.v4f16(<4 x half>, <4 x i32>, i32, i32, i32, i32)

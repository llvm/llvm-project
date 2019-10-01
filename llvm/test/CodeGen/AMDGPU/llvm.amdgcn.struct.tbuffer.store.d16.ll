; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PREGFX10,UNPACKED,PREGFX10-UNPACKED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PREGFX10,PACKED,GFX81,PREGFX10-PACKED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PREGFX10,PACKED,GFX9,PREGFX10-PACKED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,PACKED,GFX10-PACKED %s


; GCN-LABEL: {{^}}tbuffer_store_d16_x:
; GCN-DAG: s_load_dwordx4
; GCN-DAG: s_load_dword{{[x0-2]*}} s{{\[}}[[S_LO:[0-9]+]]
; GCN-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[S_LO]]
; PREGFX10: tbuffer_store_format_d16_x v[[V_LO]], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], dfmt:1, nfmt:2, 0 idxen
; GFX10: tbuffer_store_format_d16_x v[[V_LO]], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], format:33, 0 idxen
define amdgpu_kernel void @tbuffer_store_d16_x(<4 x i32> %rsrc, half %data, i32 %vindex) {
main_body:
  call void @llvm.amdgcn.struct.tbuffer.store.f16(half %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 0, i32 33, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_d16_xy:
; GCN: s_load_dword [[S_DATA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x10
; UNPACKED-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[S_DATA]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED:s[0-9]+]], [[S_DATA]], 0xffff{{$}}
; UNPACKED-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[MASKED]]
; UNPACKED-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], [[SHR]]
; PREGFX10-UNPACKED: tbuffer_store_format_d16_xy v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], dfmt:1, nfmt:2, 0 idxen

; PREGFX10-PACKED: tbuffer_store_format_d16_xy v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], dfmt:1, nfmt:2, 0 idxen
; GFX10-PACKED: tbuffer_store_format_d16_xy v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], format:33, 0 idxen
define amdgpu_kernel void @tbuffer_store_d16_xy(<4 x i32> %rsrc, <2 x half> %data, i32 %vindex) {
main_body:
  call void @llvm.amdgcn.struct.tbuffer.store.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 0, i32 33, i32 0)
  ret void
}

; GCN-LABEL: {{^}}tbuffer_store_d16_xyzw:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[S_DATA_0:[0-9]+]]:[[S_DATA_1:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x10

; UNPACKED-DAG: s_mov_b32 [[K:s[0-9]+]], 0xffff{{$}}
; UNPACKED-DAG: s_lshr_b32 [[SHR0:s[0-9]+]], s[[S_DATA_0]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED0:s[0-9]+]], s[[S_DATA_0]], [[K]]
; UNPACKED-DAG: s_lshr_b32 [[SHR1:s[0-9]+]], s[[S_DATA_1]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED1:s[0-9]+]], s[[S_DATA_1]], [[K]]

; UNPACKED-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], [[MASKED0]]
; UNPACKED-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[SHR1]]
; PREGFX10-UNPACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], dfmt:1, nfmt:2, 0 idxen

; PACKED-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], s[[S_DATA_0]]
; PACKED-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], s[[S_DATA_1]]
; PREGFX10-PACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], dfmt:1, nfmt:2, 0 idxen
; GFX10-PACKED: tbuffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], format:33, 0 idxen
define amdgpu_kernel void @tbuffer_store_d16_xyzw(<4 x i32> %rsrc, <4 x half> %data, i32 %vindex) {
main_body:
  call void @llvm.amdgcn.struct.tbuffer.store.v4f16(<4 x half> %data, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 0, i32 33, i32 0)
  ret void
}

declare void @llvm.amdgcn.struct.tbuffer.store.f16(half, <4 x i32>, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.tbuffer.store.v2f16(<2 x half>, <4 x i32>, i32, i32, i32, i32, i32)
declare void @llvm.amdgcn.struct.tbuffer.store.v4f16(<4 x half>, <4 x i32>, i32, i32, i32, i32, i32)

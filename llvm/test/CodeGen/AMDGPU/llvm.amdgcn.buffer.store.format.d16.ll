; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,UNPACKED %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx810 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,GFX81 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,PACKED,GFX9 %s

; GCN-LABEL: {{^}}buffer_store_format_d16_x:
; GCN: s_load_dword s[[LO:[0-9]+]]
; GCN: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[LO]]
; GCN: buffer_store_format_d16_x v[[V_LO]], v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
define amdgpu_kernel void @buffer_store_format_d16_x(<4 x i32> %rsrc, half %data, i32 %index) {
main_body:
  call void @llvm.amdgcn.buffer.store.format.f16(half %data, <4 x i32> %rsrc, i32 %index, i32 0, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_format_d16_xy:

; UNPACKED: s_load_dword [[S_DATA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x10
; UNPACKED-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[S_DATA]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED:s[0-9]+]], [[S_DATA]], 0xffff{{$}}
; UNPACKED-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], [[MASKED]]
; UNPACKED-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], [[SHR]]
; UNPACKED: buffer_store_format_d16_xy v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen

; PACKED: buffer_store_format_d16_xy v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
define amdgpu_kernel void @buffer_store_format_d16_xy(<4 x i32> %rsrc, <2 x half> %data, i32 %index) {
main_body:
  call void @llvm.amdgcn.buffer.store.format.v2f16(<2 x half> %data, <4 x i32> %rsrc, i32 %index, i32 0, i1 0, i1 0)
  ret void
}

; GCN-LABEL: {{^}}buffer_store_format_d16_xyzw:
; GCN-DAG: s_load_dwordx2 s{{\[}}[[S_DATA_0:[0-9]+]]:[[S_DATA_1:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x10

; UNPACKED-DAG: s_mov_b32 [[K:s[0-9]+]], 0xffff{{$}}
; UNPACKED-DAG: s_lshr_b32 [[SHR0:s[0-9]+]], s[[S_DATA_0]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED0:s[0-9]+]], s[[S_DATA_0]], [[K]]
; UNPACKED-DAG: s_lshr_b32 [[SHR1:s[0-9]+]], s[[S_DATA_1]], 16
; UNPACKED-DAG: s_and_b32 [[MASKED1:s[0-9]+]], s[[S_DATA_1]], [[K]]

; UNPACKED-DAG: v_mov_b32_e32 v[[LO:[0-9]+]], [[MASKED0]]
; UNPACKED-DAG: v_mov_b32_e32 v[[HI:[0-9]+]], [[SHR1]]

; UNPACKED: buffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen

; PACKED: v_mov_b32_e32 v[[LO:[0-9]+]], s[[S_DATA_0]]
; PACKED: v_mov_b32_e32 v[[HI:[0-9]+]], s[[S_DATA_1]]

; PACKED: buffer_store_format_d16_xyzw v{{\[}}[[LO]]:[[HI]]{{\]}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
define amdgpu_kernel void @buffer_store_format_d16_xyzw(<4 x i32> %rsrc, <4 x half> %data, i32 %index) {
main_body:
  call void @llvm.amdgcn.buffer.store.format.v4f16(<4 x half> %data, <4 x i32> %rsrc, i32 %index, i32 0, i1 0, i1 0)
  ret void
}

declare void @llvm.amdgcn.buffer.store.format.f16(half, <4 x i32>, i32, i32, i1, i1)
declare void @llvm.amdgcn.buffer.store.format.v2f16(<2 x half>, <4 x i32>, i32, i32, i1, i1)
declare void @llvm.amdgcn.buffer.store.format.v4f16(<4 x half>, <4 x i32>, i32, i32, i1, i1)

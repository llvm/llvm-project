; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs -show-mc-encoding | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED %s

; GCN-LABEL: {{^}}buffer_load_format_d16_x:
; GCN: buffer_load_format_d16_x v{{[0-9]+}}, {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
define amdgpu_ps half @buffer_load_format_d16_x(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call half @llvm.amdgcn.struct.ptr.buffer.load.format.f16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  ret half %data
}

; GCN-LABEL: {{^}}buffer_load_format_d16_xy:
; UNPACKED: buffer_load_format_d16_xy v[{{[0-9]+}}:[[HI:[0-9]+]]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: buffer_load_format_d16_xy v[[FULL:[0-9]+]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[FULL]]
define amdgpu_ps half @buffer_load_format_d16_xy(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call <2 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v2f16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %elt = extractelement <2 x half> %data, i32 1
  ret half %elt
}

; GCN-LABEL: {{^}}buffer_load_format_d16_xyz:
; UNPACKED: buffer_load_format_d16_xyz v[{{[0-9]+}}:[[HI:[0-9]+]]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: buffer_load_format_d16_xyz v[{{[0-9]+}}:[[HI:[0-9]+]]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; PACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]
define amdgpu_ps half @buffer_load_format_d16_xyz(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call <3 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v3f16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %elt = extractelement <3 x half> %data, i32 2
  ret half %elt
}

; GCN-LABEL: {{^}}buffer_load_format_d16_xyzw:
; UNPACKED: buffer_load_format_d16_xyzw v[{{[0-9]+}}:[[HI:[0-9]+]]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: buffer_load_format_d16_xyzw v[{{[0-9]+}}:[[HI:[0-9]+]]], {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[HI]]
define amdgpu_ps half @buffer_load_format_d16_xyzw(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call <4 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v4f16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %elt = extractelement <4 x half> %data, i32 3
  ret half %elt
}

; GCN-LABEL: {{^}}buffer_load_format_i16_x:
; GCN: buffer_load_format_d16_x v{{[0-9]+}}, {{v[0-9]+}}, s[{{[0-9]+:[0-9]+}}], 0 idxen
define amdgpu_ps half @buffer_load_format_i16_x(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call i16 @llvm.amdgcn.struct.ptr.buffer.load.format.i16(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %fdata = bitcast i16 %data to half
  ret half %fdata
}

declare half @llvm.amdgcn.struct.ptr.buffer.load.format.f16(ptr addrspace(8), i32, i32, i32, i32)
declare <2 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v2f16(ptr addrspace(8), i32, i32, i32, i32)
declare <3 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v3f16(ptr addrspace(8), i32, i32, i32, i32)
declare <4 x half> @llvm.amdgcn.struct.ptr.buffer.load.format.v4f16(ptr addrspace(8), i32, i32, i32, i32)
declare i16 @llvm.amdgcn.struct.ptr.buffer.load.format.i16(ptr addrspace(8), i32, i32, i32, i32)

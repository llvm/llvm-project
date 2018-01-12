; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=UNPACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx810 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED %s
; RUN: llc < %s -march=amdgcn -mcpu=gfx900 -verify-machineinstrs | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=PACKED %s

; GCN-LABEL: {{^}}buffer_load_format_d16_x:
; GCN: buffer_load_format_d16_x v{{[0-9]+}}, off, s[{{[0-9]+:[0-9]+}}], 0
define amdgpu_ps half @buffer_load_format_d16_x(<4 x i32> inreg %rsrc) {
main_body:
  %data = call half @llvm.amdgcn.buffer.load.format.f16(<4 x i32> %rsrc, i32 0, i32 0, i1 0, i1 0)
  ret half %data
}

; GCN-LABEL: {{^}}buffer_load_format_d16_xy:
; UNPACKED: buffer_load_format_d16_xy v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: buffer_load_format_d16_xy v[[FULL:[0-9]+]], off, s[{{[0-9]+:[0-9]+}}], 0
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[FULL]]
define amdgpu_ps half @buffer_load_format_d16_xy(<4 x i32> inreg %rsrc) {
main_body:
  %data = call <2 x half> @llvm.amdgcn.buffer.load.format.v2f16(<4 x i32> %rsrc, i32 0, i32 0, i1 0, i1 0)
  %elt = extractelement <2 x half> %data, i32 1
  ret half %elt
}

; GCN-LABEL: {{^}}buffer_load_format_d16_xyzw:
; UNPACKED: buffer_load_format_d16_xyzw v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0
; UNPACKED: v_mov_b32_e32 v{{[0-9]+}}, v[[HI]]

; PACKED: buffer_load_format_d16_xyzw v{{\[}}{{[0-9]+}}:[[HI:[0-9]+]]{{\]}}, off, s[{{[0-9]+:[0-9]+}}], 0
; PACKED: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v[[HI]]
define amdgpu_ps half @buffer_load_format_d16_xyzw(<4 x i32> inreg %rsrc) {
main_body:
  %data = call <4 x half> @llvm.amdgcn.buffer.load.format.v4f16(<4 x i32> %rsrc, i32 0, i32 0, i1 0, i1 0)
  %elt = extractelement <4 x half> %data, i32 3
  ret half %elt
}

declare half @llvm.amdgcn.buffer.load.format.f16(<4 x i32>, i32, i32, i1, i1)
declare <2 x half> @llvm.amdgcn.buffer.load.format.v2f16(<4 x i32>, i32, i32, i1, i1)
declare <4 x half> @llvm.amdgcn.buffer.load.format.v4f16(<4 x i32>, i32, i32, i1, i1)

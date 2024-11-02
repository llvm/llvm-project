; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefixes=R600,ALL
; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck %s --check-prefixes=GFX6,GFX678,ALL
; RUN: llc < %s -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s --check-prefixes=GFX8,GFX678,ALL
; RUN: llc < %s -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -verify-machineinstrs | FileCheck %s --check-prefixes=GFX10,GFX1011,ALL
; RUN: llc < %s -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs | FileCheck %s --check-prefixes=GFX11,GFX1011,ALL

; ALL-LABEL: {{^}}build_vector2:
; R600: MOV
; R600: MOV
; R600-NOT: MOV
; GFX678-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; GFX678-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; GFX1011-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; GFX1011-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; GFX678: buffer_store_dwordx2 v[[[X]]:[[Y]]]
; GFX10: global_store_dwordx2 v2, v[0:1], s[0:1]
; GFX11: global_store_b64 v2, v[0:1], s[0:1]
define amdgpu_kernel void @build_vector2 (<2 x i32> addrspace(1)* %out) {
entry:
  store <2 x i32> <i32 5, i32 6>, <2 x i32> addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}build_vector4:
; R600: MOV
; R600: MOV
; R600: MOV
; R600: MOV
; R600-NOT: MOV
; GFX678-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; GFX678-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; GFX678-DAG: v_mov_b32_e32 v[[Z:[0-9]]], 7
; GFX678-DAG: v_mov_b32_e32 v[[W:[0-9]]], 8
; GFX1011-DAG: v_mov_b32_e32 v[[X:[0-9]]], 5
; GFX1011-DAG: v_mov_b32_e32 v[[Y:[0-9]]], 6
; GFX1011-DAG: v_mov_b32_e32 v[[Z:[0-9]]], 7
; GFX1011-DAG: v_mov_b32_e32 v[[W:[0-9]]], 8
; GFX678: buffer_store_dwordx4 v[[[X]]:[[W]]]
; GFX10: global_store_dwordx4 v4, v[0:3], s[0:1]
; GFX11: global_store_b128 v4, v[0:3], s[0:1]
define amdgpu_kernel void @build_vector4 (<4 x i32> addrspace(1)* %out) {
entry:
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, <4 x i32> addrspace(1)* %out
  ret void
}


; ALL-LABEL: {{^}}build_vector_v2i16:
; R600: MOV
; R600-NOT: MOV
; GFX678: s_mov_b32 s3, 0xf000
; GFX678: s_mov_b32 s2, -1
; GFX678: v_mov_b32_e32 v0, 0x60005
; GFX678: s_waitcnt lgkmcnt(0)
; GFX678: buffer_store_dword v0, off, s[0:3], 0
; GFX1011: v_mov_b32_e32 v0, 0
; GFX1011: v_mov_b32_e32 v1, 0x60005
; GFX1011: s_waitcnt lgkmcnt(0)
; GFX10: global_store_dword v0, v1, s[0:1]
; GFX11: global_store_b32 v0, v1, s[0:1]
define amdgpu_kernel void @build_vector_v2i16 (<2 x i16> addrspace(1)* %out) {
entry:
  store <2 x i16> <i16 5, i16 6>, <2 x i16> addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}build_vector_v2i16_trunc:
; R600: LSHR
; R600: OR_INT
; R600: LSHR
; R600-NOT: MOV
; GFX6: s_mov_b32 s3, 0xf000
; GFX6: s_waitcnt lgkmcnt(0)
; GFX6: s_lshr_b32 s2, s2, 16
; GFX6: s_or_b32 s4, s2, 0x50000
; GFX6: s_mov_b32 s2, -1
; GFX6: v_mov_b32_e32 v0, s4
; GFX6: buffer_store_dword v0, off, s[0:3], 0
; GFX8: s_mov_b32 s3, 0xf000
; GFX8: s_mov_b32 s2, -1
; GFX8: s_waitcnt lgkmcnt(0)
; GFX8: s_lshr_b32 s4, s4, 16
; GFX8: s_or_b32 s4, s4, 0x50000
; GFX8: v_mov_b32_e32 v0, s4
; GFX8: buffer_store_dword v0, off, s[0:3], 0
; GFX1011: v_mov_b32_e32 v0, 0
; GFX1011: s_waitcnt lgkmcnt(0)
; GFX10: s_lshr_b32 s2, s2, 16
; GFX10: s_pack_ll_b32_b16 s2, s2, 5
; GFX11: s_pack_hl_b32_b16 s2, s2, 5
; GFX1011: v_mov_b32_e32 v1, s2
; GFX10: global_store_dword v0, v1, s[0:1]
; GFX11: global_store_b32 v0, v1, s[0:1]
define amdgpu_kernel void @build_vector_v2i16_trunc (<2 x i16> addrspace(1)* %out, i32 %a) {
  %srl = lshr i32 %a, 16
  %trunc = trunc i32 %srl to i16
  %ins.0 = insertelement <2 x i16> undef, i16 %trunc, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  store <2 x i16> %ins.1, <2 x i16> addrspace(1)* %out
  ret void
}

; RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s --check-prefixes=R600,ALL
; RUN: llc < %s -mtriple=amdgcn -verify-machineinstrs | FileCheck %s --check-prefixes=GFX6,GFX678,ALL
; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck %s --check-prefixes=GFX8,GFX678,ALL
; RUN: llc < %s -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -verify-machineinstrs | FileCheck %s --check-prefixes=GFX10,GFX1011,ALL
; RUN: llc < %s -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs | FileCheck %s --check-prefixes=GFX11,GFX1011,ALL
; RUN: llc < %s -mtriple=amdgcn -mcpu=gfx940 | FileCheck %s --check-prefixes=GFX940,ALL

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
define amdgpu_kernel void @build_vector2 (ptr addrspace(1) %out) {
entry:
  store <2 x i32> <i32 5, i32 6>, ptr addrspace(1) %out
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
define amdgpu_kernel void @build_vector4 (ptr addrspace(1) %out) {
entry:
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr addrspace(1) %out
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
define amdgpu_kernel void @build_vector_v2i16 (ptr addrspace(1) %out) {
entry:
  store <2 x i16> <i16 5, i16 6>, ptr addrspace(1) %out
  ret void
}

; ALL-LABEL: {{^}}build_vector_v2i16_trunc:
; R600: LSHR
; R600: OR_INT
; R600: LSHR
; R600-NOT: MOV
; GFX6: s_mov_b32 s3, 0xf000
; GFX6: s_waitcnt lgkmcnt(0)
; GFX6: v_alignbit_b32 v0, 5, s4, 16
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
define amdgpu_kernel void @build_vector_v2i16_trunc (ptr addrspace(1) %out, i32 %a) {
  %srl = lshr i32 %a, 16
  %trunc = trunc i32 %srl to i16
  %ins.0 = insertelement <2 x i16> undef, i16 %trunc, i32 0
  %ins.1 = insertelement <2 x i16> %ins.0, i16 5, i32 1
  store <2 x i16> %ins.1, ptr addrspace(1) %out
  ret void
}

; R600-LABEL: build_v2i32_from_v4i16_shuffle:
; R600:       ; %bb.0: ; %entry
; R600-NEXT:    ALU 0, @10, KC0[], KC1[]
; R600-NEXT:    TEX 1 @6
; R600-NEXT:    ALU 4, @11, KC0[CB0:0-32], KC1[]
; R600-NEXT:    MEM_RAT_CACHELESS STORE_RAW T0.XY, T1.X, 1
; R600-NEXT:    CF_END
; R600-NEXT:    PAD
; R600-NEXT:    Fetch clause starting at 6:
; R600-NEXT:     VTX_READ_16 T1.X, T0.X, 48, #3
; R600-NEXT:     VTX_READ_16 T0.X, T0.X, 44, #3
; R600-NEXT:    ALU clause starting at 10:
; R600-NEXT:     MOV * T0.X, 0.0,
; R600-NEXT:    ALU clause starting at 11:
; R600-NEXT:     LSHL * T0.Y, T1.X, literal.x,
; R600-NEXT:    16(2.242078e-44), 0(0.000000e+00)
; R600-NEXT:     LSHL T0.X, T0.X, literal.x,
; R600-NEXT:     LSHR * T1.X, KC0[2].Y, literal.y,
; R600-NEXT:    16(2.242078e-44), 2(2.802597e-45)
;
; GFX6-LABEL: build_v2i32_from_v4i16_shuffle:
; GFX6:       ; %bb.0: ; %entry
; GFX6-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x9
; GFX6-NEXT:    s_mov_b32 s7, 0xf000
; GFX6-NEXT:    s_waitcnt lgkmcnt(0)
; GFX6-NEXT:    s_lshl_b32 s3, s3, 16
; GFX6-NEXT:    s_lshl_b32 s2, s2, 16
; GFX6-NEXT:    s_mov_b32 s6, -1
; GFX6-NEXT:    s_mov_b32 s4, s0
; GFX6-NEXT:    s_mov_b32 s5, s1
; GFX6-NEXT:    v_mov_b32_e32 v0, s2
; GFX6-NEXT:    v_mov_b32_e32 v1, s3
; GFX6-NEXT:    buffer_store_dwordx2 v[0:1], off, s[4:7], 0
; GFX6-NEXT:    s_endpgm
;
; GFX8-LABEL: build_v2i32_from_v4i16_shuffle:
; GFX8:       ; %bb.0: ; %entry
; GFX8-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX8-NEXT:    s_mov_b32 s7, 0xf000
; GFX8-NEXT:    s_mov_b32 s6, -1
; GFX8-NEXT:    s_waitcnt lgkmcnt(0)
; GFX8-NEXT:    s_mov_b32 s4, s0
; GFX8-NEXT:    s_mov_b32 s5, s1
; GFX8-NEXT:    s_lshl_b32 s0, s3, 16
; GFX8-NEXT:    s_lshl_b32 s1, s2, 16
; GFX8-NEXT:    v_mov_b32_e32 v0, s1
; GFX8-NEXT:    v_mov_b32_e32 v1, s0
; GFX8-NEXT:    buffer_store_dwordx2 v[0:1], off, s[4:7], 0
; GFX8-NEXT:    s_endpgm
;
; GFX10-LABEL: build_v2i32_from_v4i16_shuffle:
; GFX10:       ; %bb.0: ; %entry
; GFX10-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x0
; GFX10-NEXT:    v_mov_b32_e32 v2, 0
; GFX10-NEXT:    s_waitcnt lgkmcnt(0)
; GFX10-NEXT:    s_lshl_b32 s2, s2, 16
; GFX10-NEXT:    s_lshl_b32 s3, s3, 16
; GFX10-NEXT:    v_mov_b32_e32 v0, s2
; GFX10-NEXT:    v_mov_b32_e32 v1, s3
; GFX10-NEXT:    global_store_dwordx2 v2, v[0:1], s[0:1]
; GFX10-NEXT:    s_endpgm
;
; GFX11-LABEL: build_v2i32_from_v4i16_shuffle:
; GFX11:       ; %bb.0: ; %entry
; GFX11-NEXT:    s_load_b128 s[0:3], s[0:1], 0x0
; GFX11-NEXT:    v_mov_b32_e32 v2, 0
; GFX11-NEXT:    s_waitcnt lgkmcnt(0)
; GFX11-NEXT:    s_lshl_b32 s2, s2, 16
; GFX11-NEXT:    s_lshl_b32 s3, s3, 16
; GFX11-NEXT:    v_mov_b32_e32 v0, s2
; GFX11-NEXT:    v_mov_b32_e32 v1, s3
; GFX11-NEXT:    global_store_b64 v2, v[0:1], s[0:1]
; GFX11-NEXT:    s_nop 0
; GFX11-NEXT:    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
; GFX11-NEXT:    s_endpgm
;
; GFX940-LABEL: build_v2i32_from_v4i16_shuffle:
; GFX940:       ; %bb.0: ; %entry
; GFX940-NEXT:    s_load_dwordx4 s[0:3], s[0:1], 0x24
; GFX940-NEXT:    v_mov_b32_e32 v2, 0
; GFX940-NEXT:    s_waitcnt lgkmcnt(0)
; GFX940-NEXT:    s_lshl_b32 s3, s3, 16
; GFX940-NEXT:    s_lshl_b32 s2, s2, 16
; GFX940-NEXT:    v_mov_b32_e32 v0, s2
; GFX940-NEXT:    v_mov_b32_e32 v1, s3
; GFX940-NEXT:    global_store_dwordx2 v2, v[0:1], s[0:1] sc0 sc1
; GFX940-NEXT:    s_endpgm
define amdgpu_kernel void @build_v2i32_from_v4i16_shuffle(ptr addrspace(1) %out, <4 x i16> %in) {
entry:
  %shuf = shufflevector <4 x i16> %in, <4 x i16> zeroinitializer, <2 x i32> <i32 0, i32 2>
  %zextended = zext <2 x i16> %shuf to <2 x i32>
  %shifted = shl <2 x i32> %zextended, <i32 16, i32 16>
  store <2 x i32> %shifted, ptr addrspace(1) %out
  ret void
}

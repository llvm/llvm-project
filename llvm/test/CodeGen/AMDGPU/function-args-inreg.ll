; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=CIGFX89,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX11 %s

define void @void_func_i1(i1 %arg0) #0 {
; CIGFX89-LABEL: void_func_i1:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    v_and_b32_e32 v0, 1, v0
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i1:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_and_b32_e32 v0, 1, v0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b8 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i1 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i1_inreg(i1 inreg %arg0) #0 {
; CIGFX89-LABEL: void_func_i1_inreg:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_and_b32 s4, s4, 1
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    v_mov_b32_e32 v0, s4
; CIGFX89-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i1_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_and_b32 s0, s0, 1
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b8 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i1 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i8(i8 %arg0) #0 {
; CIGFX89-LABEL: void_func_i8:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i8:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b8 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i8 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i8_inreg(i8 inreg %arg0) #0 {
; CIGFX89-LABEL: void_func_i8_inreg:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    v_mov_b32_e32 v0, s4
; CIGFX89-NEXT:    buffer_store_byte v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i8_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b8 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i8 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i16(i16 %arg0) #0 {
; CIGFX89-LABEL: void_func_i16:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    buffer_store_short v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i16:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b16 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i16 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i16_inreg(i16 inreg %arg0) #0 {
; CIGFX89-LABEL: void_func_i16_inreg:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    v_mov_b32_e32 v0, s4
; CIGFX89-NEXT:    buffer_store_short v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i16_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b16 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i16 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i32(i32 %arg0) #0 {
; CIGFX89-LABEL: void_func_i32:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i32 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_i32_inreg(i32 inreg %arg0) #0 {
; CIGFX89-LABEL: void_func_i32_inreg:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    v_mov_b32_e32 v0, s4
; CIGFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_i32_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store i32 %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_f16(half %arg0) #0 {
; GFX89-LABEL: void_func_f16:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    buffer_store_short v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_f16:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b16 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store half %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_f16_inreg(half inreg %arg0) #0 {
; GFX89-LABEL: void_func_f16_inreg:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    v_mov_b32_e32 v0, s4
; GFX89-NEXT:    buffer_store_short v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_f16_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b16 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store half %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_f32(float %arg0) #0 {
; CIGFX89-LABEL: void_func_f32:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_f32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store float %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_f32_inreg(float inreg %arg0) #0 {
; CIGFX89-LABEL: void_func_f32_inreg:
; CIGFX89:       ; %bb.0:
; CIGFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; CIGFX89-NEXT:    s_mov_b32 s7, 0xf000
; CIGFX89-NEXT:    s_mov_b32 s6, -1
; CIGFX89-NEXT:    v_mov_b32_e32 v0, s4
; CIGFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; CIGFX89-NEXT:    s_waitcnt vmcnt(0)
; CIGFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_f32_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store float %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_v2i16(<2 x i16> %arg0) #0 {
; GFX89-LABEL: void_func_v2i16:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_v2i16:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store <2 x i16> %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_v2i16_inreg(<2 x i16> inreg %arg0) #0 {
; GFX89-LABEL: void_func_v2i16_inreg:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    v_mov_b32_e32 v0, s4
; GFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_v2i16_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store <2 x i16> %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_v2f16(<2 x half> %arg0) #0 {
; GFX89-LABEL: void_func_v2f16:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_v2f16:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store <2 x half> %arg0, ptr addrspace(1) undef
  ret void
}

define void @void_func_v2f16_inreg(<2 x half> inreg %arg0) #0 {
; GFX89-LABEL: void_func_v2f16_inreg:
; GFX89:       ; %bb.0:
; GFX89-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX89-NEXT:    s_mov_b32 s7, 0xf000
; GFX89-NEXT:    s_mov_b32 s6, -1
; GFX89-NEXT:    v_mov_b32_e32 v0, s4
; GFX89-NEXT:    buffer_store_dword v0, off, s[4:7], 0
; GFX89-NEXT:    s_waitcnt vmcnt(0)
; GFX89-NEXT:    s_setpc_b64 s[30:31]
;
; GFX11-LABEL: void_func_v2f16_inreg:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX11-NEXT:    v_mov_b32_e32 v0, s0
; GFX11-NEXT:    s_mov_b32 s3, 0x31016000
; GFX11-NEXT:    s_mov_b32 s2, -1
; GFX11-NEXT:    buffer_store_b32 v0, off, s[0:3], 0
; GFX11-NEXT:    s_setpc_b64 s[30:31]
  store <2 x half> %arg0, ptr addrspace(1) undef
  ret void
}


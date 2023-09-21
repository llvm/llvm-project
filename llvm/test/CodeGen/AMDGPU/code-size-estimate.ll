; RUN: llc -march=amdgcn -mcpu=gfx900 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX10 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX11 %s
; RUN: llc -march=amdgcn -mcpu=gfx1150 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX11,GFX1150 %s

declare float @llvm.fabs.f32(float)
declare float @llvm.fma.f32(float, float, float)

define float @v_mul_f32_vop2(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop2:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop2:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x10]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, %y
  ret float %mul
}
; CHECK: codeLenInByte = 12

define float @v_mul_f32_vop2_inline_imm(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_inline_imm:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_inline_imm:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop2_inline_imm:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x10]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, 4.0
  ret float %mul
}
; CHECK: codeLenInByte = 12

define float @v_mul_f32_vop2_literal(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_literal:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x0a,0x00,0x00,0xf6,0x42]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_literal:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x10,0x00,0x00,0xf6,0x42]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop2_literal:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x10,0x00,0x00,0xf6,0x42]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, 123.0
  ret float %mul
}
; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0x03,0x02,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0x03,0x02,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop3_src_mods:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0x03,0x02,0x00]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, %y
  ret float %mul
}
; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods_inline_imm(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0xed,0x01,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0xed,0x01,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0xed,0x01,0x00]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 4.0
  ret float %mul
}

; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods_literal(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0x42f60000 ; encoding: [0xff,0x00,0x84,0xbe,0x00,0x00,0xf6,0x42]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, s4 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0x09,0x00,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, 0x42f60000, |v0| ; encoding: [0x00,0x02,0x08,0xd5,0xff,0x00,0x02,0x00,0x00,0x00,0xf6,0x42]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e64 v0, 0x42f60000, |v0| ; encoding: [0x00,0x02,0x08,0xd5,0xff,0x00,0x02,0x00,0x00,0x00,0xf6,0x42]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 123.0
  ret float %mul
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 20

define float @v_mul_f32_vop2_frame_index(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_frame_index:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_lshrrev_b32_e64 v1, 6, s32 ; encoding: [0x01,0x00,0x10,0xd1,0x86,0x40,0x00,0x00]
; GFX9-NEXT:    v_mul_f32_e32 v0, v1, v0 ; encoding: [0x01,0x01,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_frame_index:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_lshrrev_b32_e64 v1, 5, s32 ; encoding: [0x01,0x00,0x16,0xd5,0x85,0x40,0x00,0x00]
; GFX10-NEXT:    v_mul_f32_e32 v0, v1, v0 ; encoding: [0x01,0x01,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f32_vop2_frame_index:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_mul_f32_e32 v0, s32, v0 ; encoding: [0x20,0x00,0x00,0x10]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %alloca = alloca i32, addrspace(5)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca to i32
  %cast = bitcast i32 %ptrtoint to float
  %mul = fmul float %x, %cast
  ret float %mul
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 12

define float @v_fma_f32(float %x, float %y, float %z) {
; GFX9-LABEL: v_fma_f32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_fma_f32 v0, v0, v1, v2 ; encoding: [0x00,0x00,0xcb,0xd1,0x00,0x03,0x0a,0x04]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_fma_f32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_fma_f32 v0, v0, v1, v2 ; encoding: [0x00,0x00,0x4b,0xd5,0x00,0x03,0x0a,0x04]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_fma_f32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_fma_f32 v0, v0, v1, v2 ; encoding: [0x00,0x00,0x13,0xd6,0x00,0x03,0x0a,0x04]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float %z)
  ret float %fma
}

; CHECK: codeLenInByte = 16

define float @v_fma_f32_src_mods(float %x, float %y, float %z) {
; GFX9-LABEL: v_fma_f32_src_mods:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_fma_f32 v0, |v0|, v1, v2 ; encoding: [0x00,0x01,0xcb,0xd1,0x00,0x03,0x0a,0x04]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_fma_f32_src_mods:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_fma_f32 v0, |v0|, v1, v2 ; encoding: [0x00,0x01,0x4b,0xd5,0x00,0x03,0x0a,0x04]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_fma_f32_src_mods:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_fma_f32 v0, |v0|, v1, v2 ; encoding: [0x00,0x01,0x13,0xd6,0x00,0x03,0x0a,0x04]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fma = call float @llvm.fma.f32(float %fabs.x, float %y, float %z)
  ret float %fma
}

; CHECK: codeLenInByte = 16

define float @v_fmac_f32(float %x, float %y) {
; GFX9-LABEL: v_fmac_f32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_fma_f32 v0, v0, v1, v0 ; encoding: [0x00,0x00,0xcb,0xd1,0x00,0x03,0x02,0x04]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_fmac_f32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_fmac_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x56]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_fmac_f32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_fmac_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x56]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float %x)
  ret float %fma
}

; GFX9: codeLenInByte = 16
; GFX10: codeLenInByte = 12
; GFX11: codeLenInByte = 12

define float @v_fmaak_f32(float %x, float %y) {
; GFX9-LABEL: v_fmaak_f32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0x43800000 ; encoding: [0xff,0x00,0x84,0xbe,0x00,0x00,0x80,0x43]
; GFX9-NEXT:    v_fma_f32 v0, v0, v1, s4 ; encoding: [0x00,0x00,0xcb,0xd1,0x00,0x03,0x12,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_fmaak_f32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_fmaak_f32 v0, v0, v1, 0x43800000 ; encoding: [0x00,0x03,0x00,0x5a,0x00,0x00,0x80,0x43]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_fmaak_f32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_fmaak_f32 v0, v0, v1, 0x43800000 ; encoding: [0x00,0x03,0x00,0x5a,0x00,0x00,0x80,0x43]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 16
; GFX11: codeLenInByte = 16

define float @v_fma_k_f32_src_mods(float %x, float %y) {
; GFX9-LABEL: v_fma_k_f32_src_mods:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0x43800000 ; encoding: [0xff,0x00,0x84,0xbe,0x00,0x00,0x80,0x43]
; GFX9-NEXT:    v_fma_f32 v0, |v0|, v1, s4 ; encoding: [0x00,0x01,0xcb,0xd1,0x00,0x03,0x12,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_fma_k_f32_src_mods:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_fma_f32 v0, |v0|, v1, 0x43800000 ; encoding: [0x00,0x01,0x4b,0xd5,0x00,0x03,0xfe,0x03,0x00,0x00,0x80,0x43]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_fma_k_f32_src_mods:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_fma_f32 v0, |v0|, v1, 0x43800000 ; encoding: [0x00,0x01,0x13,0xd6,0x00,0x03,0xfe,0x03,0x00,0x00,0x80,0x43]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fma = call float @llvm.fma.f32(float %fabs.x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 20

define amdgpu_ps float @s_fmaak_f32(float inreg %x, float inreg %y) {
; GFX9-LABEL: s_fmaak_f32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    v_mov_b32_e32 v0, s1 ; encoding: [0x01,0x02,0x00,0x7e]
; GFX9-NEXT:    v_mov_b32_e32 v1, 0x43800000 ; encoding: [0xff,0x02,0x02,0x7e,0x00,0x00,0x80,0x43]
; GFX9-NEXT:    v_fma_f32 v0, s0, v0, v1 ; encoding: [0x00,0x00,0xcb,0xd1,0x00,0x00,0x06,0x04]
; GFX9-NEXT:    ; return to shader part epilog
;
; GFX10-LABEL: s_fmaak_f32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    v_mov_b32_e32 v0, 0x43800000 ; encoding: [0xff,0x02,0x00,0x7e,0x00,0x00,0x80,0x43]
; GFX10-NEXT:    v_fmac_f32_e64 v0, s0, s1 ; encoding: [0x00,0x00,0x2b,0xd5,0x00,0x02,0x00,0x00]
; GFX10-NEXT:    ; return to shader part epilog
;
; GFX1100-LABEL: s_fmaak_f32:
; GFX1100:       ; %bb.0:
; GFX1100-NEXT:    v_mov_b32_e32 v0, 0x43800000 ; encoding: [0xff,0x02,0x00,0x7e,0x00,0x00,0x80,0x43]
; GFX1100-NEXT:    s_delay_alu instid0(VALU_DEP_1) ; encoding: [0x01,0x00,0x87,0xbf]
; GFX1100-NEXT:    v_fmac_f32_e64 v0, s0, s1 ; encoding: [0x00,0x00,0x2b,0xd5,0x00,0x02,0x00,0x00]
; GFX1100-NEXT:    ; return to shader part epilog
;
; GFX1150-LABEL: s_fmaak_f32:
; GFX1150:       ; %bb.0:
; GFX1150-NEXT:    s_fmaak_f32 s0, s0, s1, 0x43800000 ; encoding: [0x00,0x01,0x80,0xa2,0x00,0x00,0x80,0x43]
; GFX1150-NEXT:    s_delay_alu instid0(SALU_CYCLE_3) ; encoding: [0x0b,0x00,0x87,0xbf]
; GFX1150-NEXT:    v_mov_b32_e32 v0, s0 ; encoding: [0x00,0x02,0x00,0x7e]
; GFX1150-NEXT:    ; return to shader part epilog
  %fma = call float @llvm.fma.f32(float %x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 16
; GFX1100: codeLenInByte = 20
; GFX1150: codeLenInByte = 16

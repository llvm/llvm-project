; RUN: llc -march=amdgcn -mcpu=gfx900 -show-mc-encoding < %s | FileCheck -check-prefixes=GFX9,NOT-GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -show-mc-encoding < %s | FileCheck -check-prefixes=GFX10,NOT-GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -show-mc-encoding < %s | FileCheck -check-prefixes=GFX11,GFX1100,NOT-GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1150 -show-mc-encoding < %s | FileCheck -check-prefixes=GFX11,GFX1150,NOT-GFX12 %s
; RUN: llc -march=amdgcn -mcpu=gfx1210 -show-mc-encoding < %s | FileCheck -check-prefixes=GFX12,GFX1210 %s

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
;
; GFX1210-LABEL: v_mul_f32_vop2:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x10]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, %y
  ret float %mul
}
; NOT-GFX12: codeLenInByte = 12
; GFX12: codeLenInByte = 28

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
;
; GFX1210-LABEL: v_mul_f32_vop2_inline_imm:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x10]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, 4.0
  ret float %mul
}
; NOT-GFX12: codeLenInByte = 12
; GFX12: codeLenInByte = 28

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
;
; GFX1210-LABEL: v_mul_f32_vop2_literal:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x10,0x00,0x00,0xf6,0x42]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul float %x, 123.0
  ret float %mul
}
; NOT-GFX12: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_mul_f32_vop3_src_mods:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0x03,0x02,0x00]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, %y
  ret float %mul
}
; NOT-GFX12: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0xed,0x01,0x00]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 4.0
  ret float %mul
}

; NOT-GFX12: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e64 v0, 0x42f60000, |v0| ; encoding: [0x00,0x02,0x08,0xd5,0xff,0x00,0x02,0x00,0x00,0x00,0xf6,0x42]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 123.0
  ret float %mul
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 20
; GFX12: codeLenInByte = 36

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
;
; GFX1210-LABEL: v_mul_f32_vop2_frame_index:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f32_e32 v0, s32, v0 ; encoding: [0x20,0x00,0x00,0x10]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %alloca = alloca i32, addrspace(5)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca to i32
  %cast = bitcast i32 %ptrtoint to float
  %mul = fmul float %x, %cast
  ret float %mul
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 12
; GFX12: codeLenInByte = 28

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
;
; GFX1210-LABEL: v_fma_f32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_fma_f32 v0, v0, v1, v2 ; encoding: [0x00,0x00,0x13,0xd6,0x00,0x03,0x0a,0x04]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float %z)
  ret float %fma
}

; NOT-GFX12: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_fma_f32_src_mods:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_fma_f32 v0, |v0|, v1, v2 ; encoding: [0x00,0x01,0x13,0xd6,0x00,0x03,0x0a,0x04]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fma = call float @llvm.fma.f32(float %fabs.x, float %y, float %z)
  ret float %fma
}

; NOT-GFX12: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_fmac_f32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_fmac_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x56]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float %x)
  ret float %fma
}

; GFX9: codeLenInByte = 16
; GFX10: codeLenInByte = 12
; GFX11: codeLenInByte = 12
; GFX12: codeLenInByte = 28

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
;
; GFX1210-LABEL: v_fmaak_f32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_fmaak_f32 v0, v0, v1, 0x43800000 ; encoding: [0x00,0x03,0x00,0x5a,0x00,0x00,0x80,0x43]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fma = call float @llvm.fma.f32(float %x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 16
; GFX11: codeLenInByte = 16
; GFX12: codeLenInByte = 32

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
;
; GFX1210-LABEL: v_fma_k_f32_src_mods:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_fma_f32 v0, |v0|, v1, 0x43800000 ; encoding: [0x00,0x01,0x13,0xd6,0x00,0x03,0xfe,0x03,0x00,0x00,0x80,0x43]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %fma = call float @llvm.fma.f32(float %fabs.x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 20
; GFX11: codeLenInByte = 20
; GFX12: codeLenInByte = 36

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
;
; GFX1210-LABEL: s_fmaak_f32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_fmaak_f32 s0, s0, s1, 0x43800000 ; encoding: [0x00,0x01,0x80,0xa2,0x00,0x00,0x80,0x43]
; GFX1210-NEXT:    s_delay_alu instid0(SALU_CYCLE_3) ; encoding: [0x0b,0x00,0x87,0xbf]
; GFX1210-NEXT:    v_mov_b32_e32 v0, s0 ; encoding: [0x00,0x02,0x00,0x7e]
; GFX1210-NEXT:    ; return to shader part epilog
  %fma = call float @llvm.fma.f32(float %x, float %y, float 256.0)
  ret float %fma
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 16
; GFX1100: codeLenInByte = 20
; GFX1150: codeLenInByte = 16
; GFX12: codeLenInByte = 16

define double @v_mul_f64_vop2_literal_32(double %x) {
; GFX9-LABEL: v_mul_f64_vop2_literal_32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0 ; encoding: [0x80,0x00,0x84,0xbe]
; GFX9-NEXT:    s_mov_b32 s5, 0x405ec000 ; encoding: [0xff,0x00,0x85,0xbe,0x00,0xc0,0x5e,0x40]
; GFX9-NEXT:    v_mul_f64 v[0:1], v[0:1], s[4:5] ; encoding: [0x00,0x00,0x81,0xd2,0x00,0x09,0x00,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f64_vop2_literal_32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    s_mov_b32 s4, 0 ; encoding: [0x80,0x03,0x84,0xbe]
; GFX10-NEXT:    s_mov_b32 s5, 0x405ec000 ; encoding: [0xff,0x03,0x85,0xbe,0x00,0xc0,0x5e,0x40]
; GFX10-NEXT:    v_mul_f64 v[0:1], v[0:1], s[4:5] ; encoding: [0x00,0x00,0x65,0xd5,0x00,0x09,0x00,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f64_vop2_literal_32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    s_mov_b32 s0, 0 ; encoding: [0x80,0x00,0x80,0xbe]
; GFX11-NEXT:    s_mov_b32 s1, 0x405ec000 ; encoding: [0xff,0x00,0x81,0xbe,0x00,0xc0,0x5e,0x40]
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1) ; encoding: [0x09,0x00,0x87,0xbf]
; GFX11-NEXT:    v_mul_f64 v[0:1], v[0:1], s[0:1] ; encoding: [0x00,0x00,0x28,0xd7,0x00,0x01,0x00,0x00]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
;
; GFX1210-LABEL: v_mul_f64_vop2_literal_32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f64_e32 v[0:1], 0x405ec000, v[0:1] ; encoding: [0xff,0x00,0x00,0x0c,0x00,0xc0,0x5e,0x40]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul double %x, 123.0
  ret double %mul
}

; GFX9: codeLenInByte = 28
; GFX10: codeLenInByte = 28
; GFX1100: codeLenInByte = 32
; GFX1150: codeLenInByte = 32
; GFX1210: codeLenInByte = 32

define double @v_mul_f64_vop2_literal_64(double %x) {
; GFX9-LABEL: v_mul_f64_vop2_literal_64:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0x66666666 ; encoding: [0xff,0x00,0x84,0xbe,0x66,0x66,0x66,0x66]
; GFX9-NEXT:    s_mov_b32 s5, 0x405ec666 ; encoding: [0xff,0x00,0x85,0xbe,0x66,0xc6,0x5e,0x40]
; GFX9-NEXT:    v_mul_f64 v[0:1], v[0:1], s[4:5] ; encoding: [0x00,0x00,0x81,0xd2,0x00,0x09,0x00,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f64_vop2_literal_64:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    s_mov_b32 s4, 0x66666666 ; encoding: [0xff,0x03,0x84,0xbe,0x66,0x66,0x66,0x66]
; GFX10-NEXT:    s_mov_b32 s5, 0x405ec666 ; encoding: [0xff,0x03,0x85,0xbe,0x66,0xc6,0x5e,0x40]
; GFX10-NEXT:    v_mul_f64 v[0:1], v[0:1], s[4:5] ; encoding: [0x00,0x00,0x65,0xd5,0x00,0x09,0x00,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_mul_f64_vop2_literal_64:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    s_mov_b32 s0, 0x66666666 ; encoding: [0xff,0x00,0x80,0xbe,0x66,0x66,0x66,0x66]
; GFX11-NEXT:    s_mov_b32 s1, 0x405ec666 ; encoding: [0xff,0x00,0x81,0xbe,0x66,0xc6,0x5e,0x40]
; GFX11-NEXT:    s_delay_alu instid0(SALU_CYCLE_1) ; encoding: [0x09,0x00,0x87,0xbf]
; GFX11-NEXT:    v_mul_f64 v[0:1], v[0:1], s[0:1] ; encoding: [0x00,0x00,0x28,0xd7,0x00,0x01,0x00,0x00]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
;
; GFX1210-LABEL: v_mul_f64_vop2_literal_64:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_mul_f64_e32 v[0:1], 0x405ec66666666666, v[0:1] ; encoding: [0xfe,0x00,0x00,0x0c,0x66,0x66,0x66,0x66,0x66,0xc6,0x5e,0x40]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %mul = fmul double %x, 123.1
  ret double %mul
}

; GFX9: codeLenInByte = 32
; GFX10: codeLenInByte = 32
; GFX1100: codeLenInByte = 36
; GFX1150: codeLenInByte = 36
; GFX1210: codeLenInByte = 36

define i64 @v_add_u64_vop2_literal_32(i64 %x) {
; GFX9-LABEL: v_add_u64_vop2_literal_32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_add_co_u32_e32 v0, vcc, 0x7b, v0 ; encoding: [0xff,0x00,0x00,0x32,0x7b,0x00,0x00,0x00]
; GFX9-NEXT:    v_addc_co_u32_e32 v1, vcc, 0, v1, vcc ; encoding: [0x80,0x02,0x02,0x38]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_add_u64_vop2_literal_32:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_add_co_u32 v0, vcc_lo, 0x7b, v0 ; encoding: [0x00,0x6a,0x0f,0xd7,0xff,0x00,0x02,0x00,0x7b,0x00,0x00,0x00]
; GFX10-NEXT:    v_add_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo ; encoding: [0x80,0x02,0x02,0x50]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_add_u64_vop2_literal_32:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_add_co_u32 v0, vcc_lo, 0x7b, v0 ; encoding: [0x00,0x6a,0x00,0xd7,0xff,0x00,0x02,0x00,0x7b,0x00,0x00,0x00]
; GFX11-NEXT:    v_add_co_ci_u32_e32 v1, vcc_lo, 0, v1, vcc_lo ; encoding: [0x80,0x02,0x02,0x40]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
;
; GFX1210-LABEL: v_add_u64_vop2_literal_32:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_add_nc_u64_e32 v[0:1], 0x7b, v[0:1] ; encoding: [0xff,0x00,0x00,0x50,0x7b,0x00,0x00,0x00]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %add = add i64 %x, 123
  ret i64 %add
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 24
; GFX1100: codeLenInByte = 24
; GFX1150: codeLenInByte = 24
; GFX1210: codeLenInByte = 32

define i64 @v_add_u64_vop2_literal_64(i64 %x) {
; GFX9-LABEL: v_add_u64_vop2_literal_64:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_add_co_u32_e32 v0, vcc, 0x12345678, v0 ; encoding: [0xff,0x00,0x00,0x32,0x78,0x56,0x34,0x12]
; GFX9-NEXT:    v_addc_co_u32_e32 v1, vcc, 1, v1, vcc ; encoding: [0x81,0x02,0x02,0x38]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_add_u64_vop2_literal_64:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_add_co_u32 v0, vcc_lo, 0x12345678, v0 ; encoding: [0x00,0x6a,0x0f,0xd7,0xff,0x00,0x02,0x00,0x78,0x56,0x34,0x12]
; GFX10-NEXT:    v_add_co_ci_u32_e32 v1, vcc_lo, 1, v1, vcc_lo ; encoding: [0x81,0x02,0x02,0x50]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
;
; GFX11-LABEL: v_add_u64_vop2_literal_64:
; GFX11:       ; %bb.0:
; GFX11-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x89,0xbf]
; GFX11-NEXT:    v_add_co_u32 v0, vcc_lo, 0x12345678, v0 ; encoding: [0x00,0x6a,0x00,0xd7,0xff,0x00,0x02,0x00,0x78,0x56,0x34,0x12]
; GFX11-NEXT:    v_add_co_ci_u32_e32 v1, vcc_lo, 1, v1, vcc_lo ; encoding: [0x81,0x02,0x02,0x40]
; GFX11-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
;
; GFX1210-LABEL: v_add_u64_vop2_literal_64:
; GFX1210:       ; %bb.0:
; GFX1210-NEXT:    s_wait_loadcnt_dscnt 0x0 ; encoding: [0x00,0x00,0xc8,0xbf]
; GFX1210-NEXT:    s_wait_expcnt 0x0 ; encoding: [0x00,0x00,0xc4,0xbf]
; GFX1210-NEXT:    s_wait_samplecnt 0x0 ; encoding: [0x00,0x00,0xc2,0xbf]
; GFX1210-NEXT:    s_wait_bvhcnt 0x0 ; encoding: [0x00,0x00,0xc3,0xbf]
; GFX1210-NEXT:    s_wait_kmcnt 0x0 ; encoding: [0x00,0x00,0xc7,0xbf]
; GFX1210-NEXT:    v_add_nc_u64_e32 v[0:1], 0x112345678, v[0:1] ; encoding: [0xfe,0x00,0x00,0x50,0x78,0x56,0x34,0x12,0x01,0x00,0x00,0x00]
; GFX1210-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x48,0x80,0xbe]
  %add = add i64 %x, 4600387192
  ret i64 %add
}

; GFX9: codeLenInByte = 20
; GFX10: codeLenInByte = 24
; GFX1100: codeLenInByte = 24
; GFX1150: codeLenInByte = 24
; GFX1210: codeLenInByte = 36

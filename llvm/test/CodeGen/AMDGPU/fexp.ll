;RUN: llc -mtriple=amdgcn-- < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=SI %s
;RUN: llc -mtriple=amdgcn-- -mcpu=fiji < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI %s
;RUN: llc -mtriple=amdgcn-- -mcpu=gfx900 < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=GFX9 %s

define float @v_exp_f32(float %arg0) {
; SI-LABEL: v_exp_f32:
; SI:       ; %bb.0:
; SI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SI-NEXT:    v_mul_f32_e32 v0, 0x3fb8aa3b, v0
; SI-NEXT:    v_exp_f32_e32 v0, v0
; SI-NEXT:    s_setpc_b64 s[30:31]
;
; VI-LABEL: v_exp_f32:
; VI:       ; %bb.0:
; VI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VI-NEXT:    v_mul_f32_e32 v0, 0x3fb8aa3b, v0
; VI-NEXT:    v_exp_f32_e32 v0, v0
; VI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_exp_f32:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mul_f32_e32 v0, 0x3fb8aa3b, v0
; GFX9-NEXT:    v_exp_f32_e32 v0, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call float @llvm.exp.f32(float %arg0)
  ret float %result
}

define <2 x float> @v_exp_v2f32(<2 x float> %arg0) {
; GCN-LABEL: v_exp_v2f32:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 [[SREG:s[0-9]+]], 0x3fb8aa3b
; GCN-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; GCN-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; GCN-NEXT:    v_exp_f32_e32 v0, v0
; GCN-NEXT:    v_exp_f32_e32 v1, v1
; GCN-NEXT:    s_setpc_b64 s[30:31]
  %result = call <2 x float> @llvm.exp.v2f32(<2 x float> %arg0)
  ret <2 x float> %result
}

define <3 x float> @v_exp_v3f32(<3 x float> %arg0) {
; GCN-LABEL: v_exp_v3f32:
; GCN:       ; %bb.0:
; GCN-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GCN-NEXT:    s_mov_b32 [[SREG:s[0-9]+]], 0x3fb8aa3b
; GCN-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; GCN-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; GCN-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; GCN-NEXT:    v_exp_f32_e32 v0, v0
; GCN-NEXT:    v_exp_f32_e32 v1, v1
; GCN-NEXT:    v_exp_f32_e32 v2, v2
; GCN-NEXT:    s_setpc_b64 s[30:31]
;
  %result = call <3 x float> @llvm.exp.v3f32(<3 x float> %arg0)
  ret <3 x float> %result
}

define <4 x float> @v_exp_v4f32(<4 x float> %arg0) {
; SI-LABEL: v_exp_v4f32:
; SI:       ; %bb.0:
; SI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SI-NEXT:    s_mov_b32 [[SREG:s[0-9]+]], 0x3fb8aa3b
; SI-NEXT:    v_mul_f32_e32 v0, [[SREG]], v0
; SI-NEXT:    v_mul_f32_e32 v1, [[SREG]], v1
; SI-NEXT:    v_mul_f32_e32 v2, [[SREG]], v2
; SI-NEXT:    v_mul_f32_e32 v3, [[SREG]], v3
; SI-NEXT:    v_exp_f32_e32 v0, v0
; SI-NEXT:    v_exp_f32_e32 v1, v1
; SI-NEXT:    v_exp_f32_e32 v2, v2
; SI-NEXT:    v_exp_f32_e32 v3, v3
; SI-NEXT:    s_setpc_b64 s[30:31]
  %result = call <4 x float> @llvm.exp.v4f32(<4 x float> %arg0)
  ret <4 x float> %result
}

define half @v_exp_f16(half %arg0) {
; SI-LABEL: v_exp_f16:
; SI:       ; %bb.0:
; SI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SI-NEXT:    v_cvt_f16_f32_e32 v0, v0
; SI-NEXT:    v_cvt_f32_f16_e32 v0, v0
; SI-NEXT:    v_mul_f32_e32 v0, 0x3fb8aa3b, v0
; SI-NEXT:    v_exp_f32_e32 v0, v0
; SI-NEXT:    s_setpc_b64 s[30:31]
;
; VI-LABEL: v_exp_f16:
; VI:       ; %bb.0:
; VI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VI-NEXT:    v_mul_f16_e32 v0, 0x3dc5, v0
; VI-NEXT:    v_exp_f16_e32 v0, v0
; VI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_exp_f16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    v_mul_f16_e32 v0, 0x3dc5, v0
; GFX9-NEXT:    v_exp_f16_e32 v0, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call half @llvm.exp.f16(half %arg0)
  ret half %result
}

define <2 x half> @v_exp_v2f16(<2 x half> %arg0) {
; SI-LABEL: v_exp_v2f16:
; SI:       ; %bb.0:
; SI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SI-NEXT:    v_cvt_f16_f32_e32 v1, v1
; SI-NEXT:    v_cvt_f16_f32_e32 v0, v0
; SI-NEXT:    s_mov_b32 [[SREG:s[0-9]+]], 0x3fb8aa3b
; SI-NEXT:    v_cvt_f32_f16_e32 v1, v1
; SI-NEXT:    v_cvt_f32_f16_e32 v0, v0
; SI-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; SI-NEXT:    v_mul_f32_e32 v{{[0-9]+}}, [[SREG]], v{{[0-9]+}}
; SI-NEXT:    v_exp_f32_e32 v0, v0
; SI-NEXT:    v_exp_f32_e32 v1, v1
; SI-NEXT:    s_setpc_b64 s[30:31]
;
; VI-LABEL: v_exp_v2f16:
; VI:       ; %bb.0:
; VI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VI-NEXT:    s_movk_i32 [[SREG:s[0-9]+]], 0x3dc5
; VI-NEXT:    v_mov_b32_e32 [[VREG:v[0-9]+]], [[SREG]]
; VI-NEXT:    v_mul_f16_sdwa [[MUL1:v[0-9]+]], v{{[0-9]+}}, [[VREG]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NEXT:    v_mul_f16_e32 [[MUL2:v[0-9]+]], [[SREG]], v{{[0-9]+}}
; VI-NEXT:    v_exp_f16_sdwa [[MUL1]], [[MUL1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
; VI-NEXT:    v_exp_f16_e32 [[MUL2]], [[MUL2]]
; VI-NEXT:    v_or_b32_e32 v{{[0-9]+}}, [[MUL2]], [[MUL1]]
; VI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_exp_v2f16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_movk_i32 [[SREG:s[0-9]+]], 0x3dc5
; GFX9-NEXT:    v_pk_mul_f16 v0, v0, [[SREG]] op_sel_hi:[1,0]
; GFX9-NEXT:    v_exp_f16_e32 v1, v0
; GFX9-NEXT:    v_exp_f16_sdwa v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1
; GFX9-NEXT:    v_pack_b32_f16 v0, v1, v0
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <2 x half> @llvm.exp.v2f16(<2 x half> %arg0)
  ret <2 x half> %result
}

; define <3 x half> @v_exp_v3f16(<3 x half> %arg0) {
;   %result = call <3 x half> @llvm.exp.v3f16(<3 x half> %arg0)
;   ret <3 x half> %result
; }

define <4 x half> @v_exp_v4f16(<4 x half> %arg0) {
; SI-LABEL: v_exp_v4f16:
; SI:       ; %bb.0:
; SI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; SI-NEXT:    v_cvt_f16_f32_e32 v3, v3
; SI-NEXT:    v_cvt_f16_f32_e32 v2, v2
; SI-NEXT:    v_cvt_f16_f32_e32 v1, v1
; SI-NEXT:    v_cvt_f16_f32_e32 v0, v0
; SI-NEXT:    s_mov_b32 [[SREG:s[0-9]+]], 0x3fb8aa3b
; SI-NEXT:    v_cvt_f32_f16_e32 v3, v3
; SI-NEXT:    v_cvt_f32_f16_e32 v2, v2
; SI-NEXT:    v_cvt_f32_f16_e32 v1, v1
; SI-NEXT:    v_cvt_f32_f16_e32 v0, v0
; SI-NEXT:    v_mul_f32_e32 v0, [[SREG]], v0
; SI-NEXT:    v_mul_f32_e32 v1, [[SREG]], v1
; SI-NEXT:    v_mul_f32_e32 v2, [[SREG]], v2
; SI-NEXT:    v_mul_f32_e32 v3, [[SREG]], v3
; SI-NEXT:    v_exp_f32_e32 v0, v0
; SI-NEXT:    v_exp_f32_e32 v1, v1
; SI-NEXT:    v_exp_f32_e32 v2, v2
; SI-NEXT:    v_exp_f32_e32 v3, v3
; SI-NEXT:    s_setpc_b64 s[30:31]
;
; VI-LABEL: v_exp_v4f16:
; VI:       ; %bb.0:
; VI-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; VI-NEXT:    s_movk_i32 [[SREG:s[0-9]+]], 0x3dc5
; VI-NEXT:    v_mov_b32_e32 [[VREG:v[0-9]+]], [[SREG]]
; VI-NEXT:    v_mul_f16_e32 [[MUL1:v[0-9]+]], [[SREG]], v1
; VI-NEXT:    v_mul_f16_e32 [[MUL2:v[0-9]+]], [[SREG]], v0
; VI-NEXT:    v_mul_f16_sdwa [[MUL3:v[0-9]+]], v1, [[VREG]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NEXT:    v_mul_f16_sdwa [[MUL4:v[0-9]+]], v0, [[VREG]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-NEXT:    v_exp_f16_e32 [[EXP1:v[0-9]+]], [[MUL1]]
; VI-NEXT:    v_exp_f16_sdwa [[EXP2:v[0-9]+]], v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
; VI-NEXT:    v_exp_f16_e32 [[EXP3:v[0-9]+]], [[MUL2]]
; VI-NEXT:    v_exp_f16_sdwa [[EXP4:v[0-9]+]], v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
; VI-NEXT:    v_or_b32_e32 v1, [[EXP1]], [[EXP2]]
; VI-NEXT:    v_or_b32_e32 v0, [[EXP3]], [[EXP4]]
; VI-NEXT:    s_setpc_b64 s[30:31]
;
; GFX9-LABEL: v_exp_v4f16:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
; GFX9-NEXT:    s_movk_i32 [[SREG:s[0-9]+]], 0x3dc5
; GFX9-NEXT:    v_mul_f16_e32 [[MUL1:v[0-9]+]], [[SREG]], v1
; GFX9-NEXT:    v_mul_f16_e32 [[MUL2:v[0-9]+]], [[SREG]], v0
; GFX9-NEXT:    v_mul_f16_sdwa [[MUL3:v[0-9]+]], v1, [[SREG]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-NEXT:    v_mul_f16_sdwa [[MUL4:v[0-9]+]], v0, [[SREG]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; GFX9-NEXT:    v_exp_f16_e32 [[EXP1:v[0-9]+]], [[MUL1]]
; GFX9-NEXT:    v_exp_f16_e32 [[EXP2:v[0-9]+]], [[MUL3]]
; GFX9-NEXT:    v_exp_f16_e32 [[EXP3:v[0-9]+]], [[MUL2]]
; GFX9-NEXT:    v_exp_f16_e32 [[EXP4:v[0-9]+]], [[MUL4]]
; GFX9-NEXT:    v_pack_b32_f16 v1, [[EXP1]], [[EXP2]]
; GFX9-NEXT:    v_pack_b32_f16 v0, [[EXP3]], [[EXP4]]
; GFX9-NEXT:    s_setpc_b64 s[30:31]
  %result = call <4 x half> @llvm.exp.v4f16(<4 x half> %arg0)
  ret <4 x half> %result
}

declare float @llvm.exp.f32(float)
declare <2 x float> @llvm.exp.v2f32(<2 x float>)
declare <3 x float> @llvm.exp.v3f32(<3 x float>)
declare <4 x float> @llvm.exp.v4f32(<4 x float>)

declare half @llvm.exp.f16(half)
declare <2 x half> @llvm.exp.v2f16(<2 x half>)
declare <3 x half> @llvm.exp.v3f16(<3 x half>)
declare <4 x half> @llvm.exp.v4f16(<4 x half>)


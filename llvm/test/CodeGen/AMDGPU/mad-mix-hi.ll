; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefix=GCN %s

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mad_mixhi_f16 v0, v0, v1, v2
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %vec.result = insertelement <2 x half> undef, half %cvt.result, i32 1
  ret <2 x half> %vec.result
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_constlo:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mov_b32_e32 v3, 0x3c00
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_constlo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %vec.result = insertelement <2 x half> <half 1.0, half undef>, half %cvt.result, i32 1
  ret <2 x half> %vec.result
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_reglo:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_reglo(half %src0, half %src1, half %src2, half %lo) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %vec = insertelement <2 x half> undef, half %lo, i32 0
  %vec.result = insertelement <2 x half> %vec, half %cvt.result, i32 1
  ret <2 x half> %vec.result
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_intpack:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v0, v0, v1, v2 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_lshlrev_b32_e32 v0, 16, v0
; GFX9-NEXT: s_setpc_b64
define i32 @v_mad_mixhi_f16_f16lo_f16lo_f16lo_intpack(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %bc = bitcast half %cvt.result to i16
  %ext = zext i16 %bc to i32
  %shr = shl i32 %ext, 16
  ret i32 %shr
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_intpack_sext:
; GFX9: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v0, v0, v1, v2 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_lshlrev_b32_e32 v0, 16, v0
; GFX9-NEXT: s_setpc_b64
define i32 @v_mad_mixhi_f16_f16lo_f16lo_f16lo_intpack_sext(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %bc = bitcast half %cvt.result to i16
  %ext = sext i16 %bc to i32
  %shr = shl i32 %ext, 16
  ret i32 %shr
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_precvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] clamp{{$}}
; GFX9-NEXT: v_cvt_f16_f32_sdwa v0, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_precvt(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %max = call float @llvm.maxnum.f32(float %result, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  %cvt.result = fptrunc float %clamp to half
  %vec.result = insertelement <2 x half> undef, half %cvt.result, i32 1
  ret <2 x half> %vec.result
}

; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_postcvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixhi_f16 v0, v0, v1, v2 op_sel_hi:[1,1,1] clamp{{$}}
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_postcvt(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  %max = call half @llvm.maxnum.f16(half %cvt.result, half 0.0)
  %clamp = call half @llvm.minnum.f16(half %max, half 1.0)
  %vec.result = insertelement <2 x half> undef, half %clamp, i32 1
  ret <2 x half> %vec.result
}


; GCN-LABEL: {{^}}v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_postcvt_multi_use:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1]{{$}}
; GFX9-NEXT: global_store_short v{{\[[0-9]+:[0-9]+\]}}, v3
; GFX9-NEXT: s_waitcnt vmcnt(0)
; GFX9-NEXT: v_mad_mixhi_f16 v0, v0, v1, v2 op_sel_hi:[1,1,1] clamp{{$}}
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mixhi_f16_f16lo_f16lo_f16lo_undeflo_clamp_postcvt_multi_use(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  store volatile half %cvt.result, half addrspace(1)* undef
  %max = call half @llvm.maxnum.f16(half %cvt.result, half 0.0)
  %clamp = call half @llvm.minnum.f16(half %max, half 1.0)
  %vec.result = insertelement <2 x half> undef, half %clamp, i32 1
  ret <2 x half> %vec.result
}

declare half @llvm.minnum.f16(half, half) #1
declare half @llvm.maxnum.f16(half, half) #1
declare float @llvm.minnum.f32(float, float) #1
declare float @llvm.maxnum.f32(float, float) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) #1

attributes #0 = { nounwind "denormal-fp-math-f32"="preserve-sign,preserve-sign" }
attributes #1 = { nounwind readnone speculatable }

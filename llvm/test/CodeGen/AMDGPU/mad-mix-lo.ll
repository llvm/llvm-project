; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -enable-misched=false < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs -enable-misched=false < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,VI %s
; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs -enable-misched=false < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,CI %s

; GCN-LABEL: mixlo_simple:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v0, v0, v1, v2{{$}}
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mac_f32_e32
; CIVI: v_cvt_f16_f32_e32
define half @mixlo_simple(float %src0, float %src1, float %src2) #0 {
  %result = call float @llvm.fmuladd.f32(float %src0, float %src1, float %src2)
  %cvt.result = fptrunc float %result to half
  ret half %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mixlo_f16_f16lo_f16lo_f16lo:
; GFX9: v_mad_mixlo_f16 v0, v0, v1, v2 op_sel_hi:[1,1,1]{{$}}
; CI: v_mac_f32
; CIVI: v_cvt_f16_f32
define half @v_mad_mixlo_f16_f16lo_f16lo_f16lo(half %src0, half %src1, half %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %src2.ext = fpext half %src2 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2.ext)
  %cvt.result = fptrunc float %result to half
  ret half %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mixlo_f16_f16lo_f16lo_f32:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v0, v0, v1, v2 op_sel_hi:[1,1,0]{{$}}
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mac_f32
define half @v_mad_mixlo_f16_f16lo_f16lo_f32(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  %cvt.result = fptrunc float %result to half
  ret half %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mixlo_f16_f16lo_f16lo_f32_clamp_post_cvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v0, v0, v1, v2 op_sel_hi:[1,1,0] clamp{{$}}
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mac_f32_e32 v{{[0-9]}}, v{{[0-9]}}, v{{[0-9]$}}
define half @v_mad_mixlo_f16_f16lo_f16lo_f32_clamp_post_cvt(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  %cvt.result = fptrunc float %result to half
  %max = call half @llvm.maxnum.f16(half %cvt.result, half 0.0)
  %clamp = call half @llvm.minnum.f16(half %max, half 1.0)
  ret half %clamp
}

; GCN-LABEL: {{^}}v_mad_mixlo_f16_f16lo_f16lo_f32_clamp_pre_cvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,0] clamp{{$}}
; GFX9-NEXT: v_cvt_f16_f32_e32 v0, v0
; GFX9-NEXT: s_setpc_b64

; CIVI: v_mac_f32_e64 v{{[0-9]}}, v{{[0-9]}}, v{{[0-9]}} clamp{{$}}
define half @v_mad_mixlo_f16_f16lo_f16lo_f32_clamp_pre_cvt(half %src0, half %src1, float %src2) #0 {
  %src0.ext = fpext half %src0 to float
  %src1.ext = fpext half %src1 to float
  %result = tail call float @llvm.fmuladd.f32(float %src0.ext, float %src1.ext, float %src2)
  %max = call float @llvm.maxnum.f32(float %result, float 0.0)
  %clamp = call float @llvm.minnum.f32(float %max, float 1.0)
  %cvt.result = fptrunc float %clamp to half
  ret half %cvt.result
}

; FIXME: Should abe able to avoid extra register because first
; operation only clobbers relevant lane.
; GCN-LABEL: {{^}}v_mad_mix_v2f32:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1]{{$}}
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1]{{$}}
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mix_v2f32(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  %cvt.result = fptrunc <2 x float> %result to <2 x half>
  ret <2 x half> %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mix_v3f32:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v1, v1, v3, v5 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixlo_f16 v3, v0, v2, v4 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <3 x half> @v_mad_mix_v3f32(<3 x half> %src0, <3 x half> %src1, <3 x half> %src2) #0 {
  %src0.ext = fpext <3 x half> %src0 to <3 x float>
  %src1.ext = fpext <3 x half> %src1 to <3 x float>
  %src2.ext = fpext <3 x half> %src2 to <3 x float>
  %result = tail call <3 x float> @llvm.fmuladd.v3f32(<3 x float> %src0.ext, <3 x float> %src1.ext, <3 x float> %src2.ext)
  %cvt.result = fptrunc <3 x float> %result to <3 x half>
  ret <3 x half> %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mix_v4f32:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v6, v1, v3, v5 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixlo_f16 v7, v0, v2, v4 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixhi_f16 v7, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixhi_f16 v6, v1, v3, v5 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mov_b32_e32 v0, v7
; GFX9-NEXT: v_mov_b32_e32 v1, v6
; GFX9-NEXT: s_setpc_b64
define <4 x half> @v_mad_mix_v4f32(<4 x half> %src0, <4 x half> %src1, <4 x half> %src2) #0 {
  %src0.ext = fpext <4 x half> %src0 to <4 x float>
  %src1.ext = fpext <4 x half> %src1 to <4 x float>
  %src2.ext = fpext <4 x half> %src2 to <4 x float>
  %result = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %src0.ext, <4 x float> %src1.ext, <4 x float> %src2.ext)
  %cvt.result = fptrunc <4 x float> %result to <4 x half>
  ret <4 x half> %cvt.result
}

; FIXME: Fold clamp
; GCN-LABEL: {{^}}v_mad_mix_v2f32_clamp_postcvt:
; GFX9: v_mad_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1] clamp{{$}}
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp{{$}}
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mix_v2f32_clamp_postcvt(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  %cvt.result = fptrunc <2 x float> %result to <2 x half>
  %max = call <2 x half> @llvm.maxnum.v2f16(<2 x half> %cvt.result, <2 x half> zeroinitializer)
  %clamp = call <2 x half> @llvm.minnum.v2f16(<2 x half> %max, <2 x half> <half 1.0, half 1.0>)
  ret <2 x half> %clamp
}

; FIXME: Should be packed into 2 registers per argument?
; GCN-LABEL: {{^}}v_mad_mix_v3f32_clamp_postcvt:
; GCN: s_waitcnt
; GFX9-DAG: v_mad_mixlo_f16 v{{[0-9]+}}, v0, v2, v4 op_sel_hi:[1,1,1] clamp
; GFX9-DAG: v_mad_mixhi_f16 v{{[0-9]+}}, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-DAG: v_mad_mixlo_f16 v{{[0-9]+}}, v1, v3, v5 op_sel_hi:[1,1,1]
; GFX9-DAG: v_pk_max_f16 v1, v1, v1 clamp
; GFX9: v_mov_b32_e32 v0, v{{[0-9]+}}
; GFX9-NEXT: s_setpc_b64
define <3 x half> @v_mad_mix_v3f32_clamp_postcvt(<3 x half> %src0, <3 x half> %src1, <3 x half> %src2) #0 {
  %src0.ext = fpext <3 x half> %src0 to <3 x float>
  %src1.ext = fpext <3 x half> %src1 to <3 x float>
  %src2.ext = fpext <3 x half> %src2 to <3 x float>
  %result = tail call <3 x float> @llvm.fmuladd.v3f32(<3 x float> %src0.ext, <3 x float> %src1.ext, <3 x float> %src2.ext)
  %cvt.result = fptrunc <3 x float> %result to <3 x half>
  %max = call <3 x half> @llvm.maxnum.v3f16(<3 x half> %cvt.result, <3 x half> zeroinitializer)
  %clamp = call <3 x half> @llvm.minnum.v3f16(<3 x half> %max, <3 x half> <half 1.0, half 1.0, half 1.0>)
  ret <3 x half> %clamp
}

; GCN-LABEL: {{^}}v_mad_mix_v4f32_clamp_postcvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v6, v0, v2, v4 op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mixhi_f16 v6, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mixlo_f16 v2, v1, v3, v5 op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mixhi_f16 v2, v1, v3, v5 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mov_b32_e32 v0, v6
; GFX9-NEXT: v_mov_b32_e32 v1, v2
; GFX9-NEXT: s_setpc_b64
define <4 x half> @v_mad_mix_v4f32_clamp_postcvt(<4 x half> %src0, <4 x half> %src1, <4 x half> %src2) #0 {
  %src0.ext = fpext <4 x half> %src0 to <4 x float>
  %src1.ext = fpext <4 x half> %src1 to <4 x float>
  %src2.ext = fpext <4 x half> %src2 to <4 x float>
  %result = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %src0.ext, <4 x float> %src1.ext, <4 x float> %src2.ext)
  %cvt.result = fptrunc <4 x float> %result to <4 x half>
  %max = call <4 x half> @llvm.maxnum.v4f16(<4 x half> %cvt.result, <4 x half> zeroinitializer)
  %clamp = call <4 x half> @llvm.minnum.v4f16(<4 x half> %max, <4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>)
  ret <4 x half> %clamp
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_clamp_postcvt_lo:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mix_v2f32_clamp_postcvt_lo(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  %cvt.result = fptrunc <2 x float> %result to <2 x half>
  %cvt.lo = extractelement <2 x half> %cvt.result, i32 0
  %max.lo = call half @llvm.maxnum.f16(half %cvt.lo, half 0.0)
  %clamp.lo = call half @llvm.minnum.f16(half %max.lo, half 1.0)
  %insert = insertelement <2 x half> %cvt.result, half %clamp.lo, i32 0
  ret <2 x half> %insert
}

; GCN-LABEL: {{^}}v_mad_mix_v2f32_clamp_postcvt_hi:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mixlo_f16 v3, v0, v1, v2 op_sel_hi:[1,1,1]
; GFX9-NEXT: v_mad_mixhi_f16 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mov_b32_e32 v0, v3
; GFX9-NEXT: s_setpc_b64
define <2 x half> @v_mad_mix_v2f32_clamp_postcvt_hi(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  %cvt.result = fptrunc <2 x float> %result to <2 x half>
  %cvt.hi = extractelement <2 x half> %cvt.result, i32 1
  %max.hi = call half @llvm.maxnum.f16(half %cvt.hi, half 0.0)
  %clamp.hi = call half @llvm.minnum.f16(half %max.hi, half 1.0)
  %insert = insertelement <2 x half> %cvt.result, half %clamp.hi, i32 1
  ret <2 x half> %insert
}

; FIXME: Should be able to use mixlo/mixhi
; GCN-LABEL: {{^}}v_mad_mix_v2f32_clamp_precvt:
; GFX9: v_mad_mix_f32 v3, v0, v1, v2 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mix_f32 v0, v0, v1, v2 op_sel_hi:[1,1,1] clamp
; GFX9: v_cvt_f16_f32_e32 v0, v0
; GFX9: v_cvt_f16_f32_e32 v1, v3
; GFX9: v_and_b32_e32 v0, 0xffff, v0
; GFX9: v_lshl_or_b32 v0, v1, 16, v0
; GFX9: s_setpc_b64
define <2 x half> @v_mad_mix_v2f32_clamp_precvt(<2 x half> %src0, <2 x half> %src1, <2 x half> %src2) #0 {
  %src0.ext = fpext <2 x half> %src0 to <2 x float>
  %src1.ext = fpext <2 x half> %src1 to <2 x float>
  %src2.ext = fpext <2 x half> %src2 to <2 x float>
  %result = tail call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %src0.ext, <2 x float> %src1.ext, <2 x float> %src2.ext)
  %max = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %result, <2 x float> zeroinitializer)
  %clamp = call <2 x float> @llvm.minnum.v2f32(<2 x float> %max, <2 x float> <float 1.0, float 1.0>)
  %cvt.result = fptrunc <2 x float> %clamp to <2 x half>
  ret <2 x half> %cvt.result
}

; FIXME: Handling undef 4th component
; GCN-LABEL: {{^}}v_mad_mix_v3f32_clamp_precvt:
; GCN: s_waitcnt
; GFX9-NEXT: v_mad_mix_f32 v1, v1, v3, v5 op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mix_f32 v3, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_mad_mix_f32 v0, v0, v2, v4 op_sel_hi:[1,1,1] clamp
; GFX9-NEXT: v_cvt_f16_f32_e32 v0, v0
; GFX9-NEXT: v_cvt_f16_f32_e32 v2, v3
; GFX9-NEXT: v_cvt_f16_f32_e32 v1, v1
; GFX9-NEXT: v_and_b32_e32 v0, 0xffff, v0
; GFX9-NEXT: v_lshl_or_b32 v0, v2, 16, v0
; GFX9-NEXT: s_setpc_b64
define <3 x half> @v_mad_mix_v3f32_clamp_precvt(<3 x half> %src0, <3 x half> %src1, <3 x half> %src2) #0 {
  %src0.ext = fpext <3 x half> %src0 to <3 x float>
  %src1.ext = fpext <3 x half> %src1 to <3 x float>
  %src2.ext = fpext <3 x half> %src2 to <3 x float>
  %result = tail call <3 x float> @llvm.fmuladd.v3f32(<3 x float> %src0.ext, <3 x float> %src1.ext, <3 x float> %src2.ext)
  %max = call <3 x float> @llvm.maxnum.v3f32(<3 x float> %result, <3 x float> zeroinitializer)
  %clamp = call <3 x float> @llvm.minnum.v3f32(<3 x float> %max, <3 x float> <float 1.0, float 1.0, float 1.0>)
  %cvt.result = fptrunc <3 x float> %clamp to <3 x half>
  ret <3 x half> %cvt.result
}

; GCN-LABEL: {{^}}v_mad_mix_v4f32_clamp_precvt:
; GFX9: v_mad_mix_f32 v6, v1, v3, v5 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9: v_mad_mix_f32 v1, v1, v3, v5 op_sel_hi:[1,1,1] clamp
; GFX9: v_mad_mix_f32 v3, v0, v2, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1] clamp
; GFX9: v_mad_mix_f32 v0, v0, v2, v4 op_sel_hi:[1,1,1] clamp

; GFX9: v_cvt_f16_f32
; GFX9: v_cvt_f16_f32
; GFX9: v_cvt_f16_f32
; GFX9: v_cvt_f16_f32
define <4 x half> @v_mad_mix_v4f32_clamp_precvt(<4 x half> %src0, <4 x half> %src1, <4 x half> %src2) #0 {
  %src0.ext = fpext <4 x half> %src0 to <4 x float>
  %src1.ext = fpext <4 x half> %src1 to <4 x float>
  %src2.ext = fpext <4 x half> %src2 to <4 x float>
  %result = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %src0.ext, <4 x float> %src1.ext, <4 x float> %src2.ext)
  %max = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %result, <4 x float> zeroinitializer)
  %clamp = call <4 x float> @llvm.minnum.v4f32(<4 x float> %max, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  %cvt.result = fptrunc <4 x float> %clamp to <4 x half>
  ret <4 x half> %cvt.result
}

declare half @llvm.minnum.f16(half, half) #1
declare <2 x half> @llvm.minnum.v2f16(<2 x half>, <2 x half>) #1
declare <3 x half> @llvm.minnum.v3f16(<3 x half>, <3 x half>) #1
declare <4 x half> @llvm.minnum.v4f16(<4 x half>, <4 x half>) #1

declare half @llvm.maxnum.f16(half, half) #1
declare <2 x half> @llvm.maxnum.v2f16(<2 x half>, <2 x half>) #1
declare <3 x half> @llvm.maxnum.v3f16(<3 x half>, <3 x half>) #1
declare <4 x half> @llvm.maxnum.v4f16(<4 x half>, <4 x half>) #1

declare float @llvm.minnum.f32(float, float) #1
declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>) #1
declare <3 x float> @llvm.minnum.v3f32(<3 x float>, <3 x float>) #1
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>) #1

declare float @llvm.maxnum.f32(float, float) #1
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #1
declare <3 x float> @llvm.maxnum.v3f32(<3 x float>, <3 x float>) #1
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>) #1

declare float @llvm.fmuladd.f32(float, float, float) #1
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) #1
declare <3 x float> @llvm.fmuladd.v3f32(<3 x float>, <3 x float>, <3 x float>) #1
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }

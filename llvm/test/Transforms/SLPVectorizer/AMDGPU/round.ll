; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX7 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX8 %s

; GCN-LABEL: @round_v2f16(
; GFX7: call half @llvm.round.f16(
; GFX7: call half @llvm.round.f16(

; GFX8: call <2 x half> @llvm.round.v2f16(
define <2 x half> @round_v2f16(<2 x half> %arg) {
bb:
  %tmp = extractelement <2 x half> %arg, i64 0
  %tmp1 = tail call half @llvm.round.half(half %tmp)
  %tmp2 = insertelement <2 x half> undef, half %tmp1, i64 0
  %tmp3 = extractelement <2 x half> %arg, i64 1
  %tmp4 = tail call half @llvm.round.half(half %tmp3)
  %tmp5 = insertelement <2 x half> %tmp2, half %tmp4, i64 1
  ret <2 x half> %tmp5
}

; GCN-LABEL: @round_v2f32(
; GCN: call float @llvm.round.f32(
; GCN: call float @llvm.round.f32(
define <2 x float> @round_v2f32(<2 x float> %arg) {
bb:
  %tmp = extractelement <2 x float> %arg, i64 0
  %tmp1 = tail call float @llvm.round.f32(float %tmp)
  %tmp2 = insertelement <2 x float> undef, float %tmp1, i64 0
  %tmp3 = extractelement <2 x float> %arg, i64 1
  %tmp4 = tail call float @llvm.round.f32(float %tmp3)
  %tmp5 = insertelement <2 x float> %tmp2, float %tmp4, i64 1
  ret <2 x float> %tmp5
}

declare half @llvm.round.half(half) #0
declare float @llvm.round.f32(float) #0

attributes #0 = { nounwind readnone speculatable willreturn }

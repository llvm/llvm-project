; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX7 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -slp-vectorizer %s | FileCheck -check-prefixes=GCN,GFX8 %s

; GCN-LABEL: @bswap_v2i16(
; GFX7: call i16 @llvm.bswap.i16(
; GFX7: call i16 @llvm.bswap.i16(

; GFX8: call <2 x i16> @llvm.bswap.v2i16(
define <2 x i16> @bswap_v2i16(<2 x i16> %arg) {
bb:
  %tmp = extractelement <2 x i16> %arg, i64 0
  %tmp1 = tail call i16 @llvm.bswap.i16(i16 %tmp)
  %tmp2 = insertelement <2 x i16> undef, i16 %tmp1, i64 0
  %tmp3 = extractelement <2 x i16> %arg, i64 1
  %tmp4 = tail call i16 @llvm.bswap.i16(i16 %tmp3)
  %tmp5 = insertelement <2 x i16> %tmp2, i16 %tmp4, i64 1
  ret <2 x i16> %tmp5
}

; GCN-LABEL: @bswap_v2i32(
; GCN: call i32 @llvm.bswap.i32
; GCN: call i32 @llvm.bswap.i32
define <2 x i32> @bswap_v2i32(<2 x i32> %arg) {
bb:
  %tmp = extractelement <2 x i32> %arg, i64 0
  %tmp1 = tail call i32 @llvm.bswap.i32(i32 %tmp)
  %tmp2 = insertelement <2 x i32> undef, i32 %tmp1, i64 0
  %tmp3 = extractelement <2 x i32> %arg, i64 1
  %tmp4 = tail call i32 @llvm.bswap.i32(i32 %tmp3)
  %tmp5 = insertelement <2 x i32> %tmp2, i32 %tmp4, i64 1
  ret <2 x i32> %tmp5
}

declare i16 @llvm.bswap.i16(i16) #0
declare i32 @llvm.bswap.i32(i32) #0

attributes #0 = { nounwind readnone speculatable willreturn }

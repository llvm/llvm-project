; RUN: opt -march=hexagon -hexagon-vc -S < %s | FileCheck %s

; Test that the HexagonVectorCombine pass identifies instruction
; pairs whose difference in pointers is zero. This creates a vector
; load to handle masked and unmasked loads that have no base
; pointer difference and replaces the masked and unmasked loads
; with selects

; CHECK: select

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define dllexport void @f0(i8** %a0) local_unnamed_addr #0 {
b0:
  %v0 = load i8*, i8** %a0, align 4
  %v1 = getelementptr i8, i8* %v0, i32 1794
  %v2 = bitcast i8* %v1 to <64 x i16>*
  call void @llvm.assume(i1 true) [ "align"(i8* %v0, i32 128) ]
  %v3 = load <64 x i16>, <64 x i16>* %v2, align 128
  %v4 = add <64 x i16> %v3, %v3
  call void @llvm.assume(i1 true) [ "align"(i8* %v0, i32 128) ]
  %v5 = tail call <64 x i16> @llvm.masked.load.v64i16.p0v64i16(<64 x i16>* %v2, i32 128, <64 x i1> <i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true>, <64 x i16> undef)
  call void @llvm.assume(i1 true) [ "align"(i8* %v0, i32 128) ]
  %v6 = add <64 x i16> %v4, %v5
  store <64 x i16> %v6, <64 x i16>* %v2, align 128
  ret void
}

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: argmemonly nofree nosync nounwind readonly willreturn
declare <64 x i16> @llvm.masked.load.v64i16.p0v64i16(<64 x i16>*, i32 immarg, <64 x i1>, <64 x i16>) #2

attributes #0 = { "target-features"="+hvxv68,+hvx-length128b,+hvxv68,+hvx-length128b" }
attributes #1 = { nofree nosync nounwind willreturn }
attributes #2 = { argmemonly nofree nosync nounwind readonly willreturn }

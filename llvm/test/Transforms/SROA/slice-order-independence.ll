; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

; Check that the chosen type for a split is independent from the order of
; slices even in case of types that are skipped because their width is not a
; byte width multiple
define void @skipped_inttype_first({ i16*, i32 }*) {
; CHECK-LABEL: @skipped_inttype_first
; CHECK: alloca i8*
  %arg = alloca { i16*, i32 }, align 8
  %2 = bitcast { i16*, i32 }* %0 to i8*
  %3 = bitcast { i16*, i32 }* %arg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %3, i8* align 8 %2, i32 16, i1 false)
  %b = getelementptr inbounds { i16*, i32 }, { i16*, i32 }* %arg, i64 0, i32 0
  %pb0 = bitcast i16** %b to i63*
  %b0 = load i63, i63* %pb0
  %pb1 = bitcast i16** %b to i8**
  %b1 = load i8*, i8** %pb1
  ret void
}

define void @skipped_inttype_last({ i16*, i32 }*) {
; CHECK-LABEL: @skipped_inttype_last
; CHECK: alloca i8*
  %arg = alloca { i16*, i32 }, align 8
  %2 = bitcast { i16*, i32 }* %0 to i8*
  %3 = bitcast { i16*, i32 }* %arg to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %3, i8* align 8 %2, i32 16, i1 false)
  %b = getelementptr inbounds { i16*, i32 }, { i16*, i32 }* %arg, i64 0, i32 0
  %pb1 = bitcast i16** %b to i8**
  %b1 = load i8*, i8** %pb1
  %pb0 = bitcast i16** %b to i63*
  %b0 = load i63, i63* %pb0
  ret void
}

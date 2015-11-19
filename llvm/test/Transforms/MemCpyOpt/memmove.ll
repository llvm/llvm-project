; RUN: opt < %s -basicaa -memcpyopt -S | FileCheck %s
; These memmoves should get optimized to memcpys.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.0"

declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

define i8* @test1(i8* nocapture %src) nounwind {
entry:
; CHECK-LABEL: @test1(
; CHECK: call void @llvm.memcpy

  %malloccall = tail call i8* @malloc(i32 trunc (i64 mul nuw (i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64), i64 13) to i32))
  %call3 = bitcast i8* %malloccall to [13 x i8]*
  %call3.sub = getelementptr inbounds [13 x i8], [13 x i8]* %call3, i64 0, i64 0
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %call3.sub, i8* %src, i64 13, i32 1, i1 false)
  ret i8* %call3.sub
}
declare noalias i8* @malloc(i32)


define void @test2(i8* %P) nounwind {
entry:
; CHECK-LABEL: @test2(
; CHECK: call void @llvm.memcpy
  %add.ptr = getelementptr i8, i8* %P, i64 16
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %P, i8* %add.ptr, i64 16, i32 1, i1 false)
  ret void
}

; This cannot be optimize because the src/dst really do overlap.
define void @test3(i8* %P) nounwind {
entry:
; CHECK-LABEL: @test3(
; CHECK: call void @llvm.memmove
  %add.ptr = getelementptr i8, i8* %P, i64 16
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %P, i8* %add.ptr, i64 17, i32 1, i1 false)
  ret void
}

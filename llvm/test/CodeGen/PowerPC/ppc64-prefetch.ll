target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc < %s | FileCheck %s

define void @test1(i8* %a, ...) nounwind {
entry:
  call void @llvm.prefetch(i8* %a, i32 0, i32 3, i32 1)
  ret void
}

declare void @llvm.prefetch(i8*, i32, i32, i32)

; CHECK: @test1
; CHECK: dcbt


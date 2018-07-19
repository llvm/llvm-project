; RUN: opt < %s -globalopt -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@G = internal global i32* null

define void @t() {
; CHECK: @t()
; CHECK-NOT: call i8* @malloc
; CHECK-NEXT: ret void
  %malloccall = tail call i8* @malloc(i64 mul (i64 100, i64 4))
  %P = bitcast i8* %malloccall to i32*
  store i32* %P, i32** @G
  %GV = load i32*, i32** @G
  %GVe = getelementptr i32, i32* %GV, i32 40
  store i32 20, i32* %GVe
  ret void
}

declare noalias i8* @malloc(i64)

; PR6422
; RUN: opt -passes=globalopt -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@fixLRBT = internal global i32* null              ; <i32**> [#uses=2]

declare noalias i8* @malloc(i32)

define i32 @parser() nounwind {
bb918:
  %malloccall.i10 = call i8* @malloc(i32 16) nounwind ; <i8*> [#uses=1]
  %0 = bitcast i8* %malloccall.i10 to i32*        ; <i32*> [#uses=1]
  store i32* %0, i32** @fixLRBT, align 8
  %1 = load i32*, i32** @fixLRBT, align 8               ; <i32*> [#uses=0]
  %A = load i32, i32* %1
  ret i32 %A
}

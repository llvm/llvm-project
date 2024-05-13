; PR6422
; RUN: opt -passes=globalopt -S < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@fixLRBT = internal global ptr null              ; <ptr> [#uses=2]

declare noalias ptr @malloc(i32)

define i32 @parser() nounwind {
bb918:
  %malloccall.i10 = call ptr @malloc(i32 16) nounwind ; <ptr> [#uses=1]
  store ptr %malloccall.i10, ptr @fixLRBT, align 8
  %0 = load ptr, ptr @fixLRBT, align 8               ; <ptr> [#uses=0]
  %A = load i32, ptr %0
  ret i32 %A
}

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis
; PR3611

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

@f = alias void ()* @0		; <void ()*> [#uses=0]
@g = alias void ()* @1		; <void ()*> [#uses=0]
@h = external global void ()* 		; <void ()*> [#uses=0]

define internal void @0() nounwind {
entry:
  store void()* @0, void()** @h
	br label %return

return:		; preds = %entry
	ret void
}

define internal void @1() nounwind {
entry:
	br label %return

return:		; preds = %entry
	ret void
}

; RUN: llc < %s -march=x86
; PR4188
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
@g_9 = external global i32		; <i32*> [#uses=1]

define i32 @int86(i32 %p_87) nounwind {
entry:
	%0 = trunc i32 %p_87 to i8		; <i8> [#uses=1]
	%1 = icmp ne i8 %0, 0		; <i1> [#uses=1]
	br label %bb

bb:		; preds = %bb.i, %bb, %entry
	%2 = load volatile i32, i32* @g_9, align 4		; <i32> [#uses=2]
	%3 = icmp sgt i32 %2, 1		; <i1> [#uses=1]
	%4 = and i1 %3, %1		; <i1> [#uses=1]
	br i1 %4, label %bb.i, label %bb

bb.i:		; preds = %bb
	%5 = icmp sgt i32 0, %2		; <i1> [#uses=0]
	br label %bb
}

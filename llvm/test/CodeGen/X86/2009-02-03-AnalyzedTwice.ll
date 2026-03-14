; RUN: llc < %s
; PR3411

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@g_3 = external dso_local global i32		; <ptr> [#uses=1]

define void @bar(i64 %p_66) nounwind {
entry:
	br i1 false, label %bb, label %bb1

bb:		; preds = %entry
	unreachable

bb1:		; preds = %entry
	%0 = load i32, ptr @g_3, align 4		; <i32> [#uses=2]
	%1 = sext i32 %0 to i64		; <i64> [#uses=1]
	%2 = or i64 %1, %p_66		; <i64> [#uses=1]
	%3 = shl i64 %2, 0		; <i64> [#uses=1]
	%4 = and i64 %3, %p_66		; <i64> [#uses=1]
	%5 = icmp eq i64 %4, 1		; <i1> [#uses=1]
	%6 = trunc i64 %p_66 to i32		; <i32> [#uses=2]
	%7 = or i32 %0, %6		; <i32> [#uses=2]
	%8 = sub i32 %7, %6		; <i32> [#uses=1]
	%iftmp.0.0 = select i1 %5, i32 %8, i32 %7		; <i32> [#uses=1]
	%9 = tail call i32 @foo(i32 %iftmp.0.0) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @foo(i32)

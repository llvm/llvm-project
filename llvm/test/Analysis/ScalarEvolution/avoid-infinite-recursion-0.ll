; RUN: opt < %s -passes='print<scalar-evolution>'
; PR4537

; ModuleID = 'b.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test() {
entry:
	%0 = load ptr, ptr undef, align 8		; <ptr> [#uses=1]
	%1 = ptrtoint ptr %0 to i64		; <i64> [#uses=1]
	%2 = sub i64 undef, %1		; <i64> [#uses=1]
	%3 = lshr i64 %2, 3		; <i64> [#uses=1]
	%4 = trunc i64 %3 to i32		; <i32> [#uses=2]
	br i1 undef, label %bb10, label %bb4.i

bb4.i:		; preds = %bb4.i, %entry
	%i.0.i6 = phi i32 [ %8, %bb4.i ], [ 0, %entry ]		; <i32> [#uses=2]
	%5 = sub i32 %4, %i.0.i6		; <i32> [#uses=1]
	%6 = sext i32 %5 to i64		; <i64> [#uses=1]
	%7 = udiv i64 undef, %6		; <i64> [#uses=1]
	%8 = add i32 %i.0.i6, 1		; <i32> [#uses=2]
	%phitmp = icmp eq i64 %7, 0		; <i1> [#uses=1]
	%.not.i = icmp sge i32 %8, %4		; <i1> [#uses=1]
	%or.cond.i = or i1 %phitmp, %.not.i		; <i1> [#uses=1]
	br i1 %or.cond.i, label %bb10, label %bb4.i

bb10:		; preds = %bb4.i, %entry
	unreachable
}

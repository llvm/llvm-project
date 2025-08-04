; RUN: opt < %s -passes=indvars
; PR4436

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define ptr @string_expandtabs(i32 %n, ptr %m, i1 %arg) nounwind {
entry:
	br i1 %arg, label %bb33, label %bb1

bb1:		; preds = %entry
	br i1 %arg, label %overflow1, label %bb15

bb15:		; preds = %bb1
	br i1 %arg, label %bb33, label %bb17

bb17:		; preds = %bb15
	br label %bb30

bb19:		; preds = %bb30
	br i1 %arg, label %bb20, label %bb29

bb20:		; preds = %bb19
	%0 = load i32, ptr undef, align 4		; <i32> [#uses=1]
	%1 = sub i32 %0, %n		; <i32> [#uses=1]
	br label %bb23

bb21:		; preds = %bb23
	%2 = icmp ult ptr %q.0, %m		; <i1> [#uses=1]
	br i1 %2, label %bb22, label %overflow2

bb22:		; preds = %bb21
	%3 = getelementptr i8, ptr %q.0, i32 1		; <ptr> [#uses=1]
	br label %bb23

bb23:		; preds = %bb22, %bb20
	%i.2 = phi i32 [ %1, %bb20 ], [ %4, %bb22 ]		; <i32> [#uses=1]
	%q.0 = phi ptr [ undef, %bb20 ], [ %3, %bb22 ]		; <ptr> [#uses=3]
	%4 = add i32 %i.2, -1		; <i32> [#uses=2]
	%5 = icmp eq i32 %4, -1		; <i1> [#uses=1]
	br i1 %5, label %bb29, label %bb21

bb29:		; preds = %bb23, %bb19
	%q.1 = phi ptr [ undef, %bb19 ], [ %q.0, %bb23 ]		; <ptr> [#uses=0]
	br label %bb30

bb30:		; preds = %bb29, %bb17
	br i1 %arg, label %bb19, label %bb33

overflow2:		; preds = %bb21
	br i1 %arg, label %bb32, label %overflow1

bb32:		; preds = %overflow2
	br label %overflow1

overflow1:		; preds = %bb32, %overflow2, %bb1
	ret ptr null

bb33:		; preds = %bb30, %bb15, %entry
	ret ptr undef
}

; RUN: llc < %s -mtriple=i386-apple-darwin | grep testb

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.x = type <{ i8, i8, i16 }>

define i32 @foo(%struct.x* %p) nounwind {
entry:
	%0 = getelementptr %struct.x* %p, i32 0, i32 0		; <i8*> [#uses=1]
	store i8 55, i8* %0, align 1
	%1 = bitcast %struct.x* %p to i32*		; <i32*> [#uses=1]
	%2 = load i32* %1, align 1		; <i32> [#uses=1]
	%3 = and i32 %2, 512		; <i32> [#uses=1]
	%4 = icmp eq i32 %3, 0		; <i1> [#uses=1]
	br i1 %4, label %bb5, label %bb

bb:		; preds = %entry
	%5 = tail call i32 (...)* @xx() nounwind		; <i32> [#uses=1]
	ret i32 %5

bb5:		; preds = %entry
	ret i32 0
}

declare i32 @xx(...)

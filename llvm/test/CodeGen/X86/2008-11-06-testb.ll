; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s

; CHECK: testb

; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
	%struct.x = type <{ i8, i8, i16 }>

define i32 @foo(ptr %p) nounwind {
entry:
	%0 = getelementptr %struct.x, ptr %p, i32 0, i32 0		; <ptr> [#uses=1]
	store i8 55, ptr %0, align 1
	%1 = load i32, ptr %p, align 1		; <i32> [#uses=1]
	%2 = and i32 %1, 512		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb5, label %bb

bb:		; preds = %entry
	%4 = tail call i32 (...) @xx() nounwind		; <i32> [#uses=1]
	ret i32 %4

bb5:		; preds = %entry
	ret i32 0
}

declare i32 @xx(...)

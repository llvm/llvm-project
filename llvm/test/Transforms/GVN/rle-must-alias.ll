; RUN: opt < %s -basicaa -gvn -S | FileCheck %s

; GVN should eliminate the fully redundant %9 GEP which 
; allows DEAD to be removed.  This is PR3198.

; The %7 and %4 loads combine to make %DEAD unneeded.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
@H = common global [100 x i32] zeroinitializer, align 32		; <[100 x i32]*> [#uses=3]
@G = common global i32 0		; <i32*> [#uses=2]

define i32 @test(i32 %i) nounwind {
entry:
	%0 = tail call i32 (...) @foo() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
	%2 = tail call i32 (...) @bar() nounwind		; <i32> [#uses=0]
	%3 = getelementptr [100 x i32], [100 x i32]* @H, i32 0, i32 %i		; <i32*> [#uses=1]
	%4 = load i32, i32* %3, align 4		; <i32> [#uses=1]
	store i32 %4, i32* @G, align 4
	br label %bb3

bb1:		; preds = %entry
	%5 = tail call i32 (...) @baz() nounwind		; <i32> [#uses=0]
	%6 = getelementptr [100 x i32], [100 x i32]* @H, i32 0, i32 %i		; <i32*> [#uses=1]
	%7 = load i32, i32* %6, align 4		; <i32> [#uses=2]
	store i32 %7, i32* @G, align 4
	%8 = icmp eq i32 %7, 0		; <i1> [#uses=1]
	br i1 %8, label %bb3, label %bb4

bb3:		; preds = %bb1, %bb
	%9 = getelementptr [100 x i32], [100 x i32]* @H, i32 0, i32 %i		; <i32*> [#uses=1]
	%DEAD = load i32, i32* %9, align 4		; <i32> [#uses=1]
; CHECK: %DEAD = phi i32 [ 0, %bb1 ], [ %4, %bb ]
	ret i32 %DEAD

bb4:		; preds = %bb1
	ret i32 0
}

declare i32 @foo(...)

declare i32 @bar(...)

declare i32 @baz(...)

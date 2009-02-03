; RUN: llvm-as < %s | opt -jump-threading -mem2reg -simplifycfg | llvm-dis | grep {ret i32 1}
; rdar://6402033

; Test that we can thread through the block with the partially redundant load (%2).
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @foo(i32* %P) nounwind {
entry:
	%0 = tail call i32 (...)* @f1() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
	store i32 42, i32* %P, align 4
	br label %bb1

bb1:		; preds = %entry, %bb
	%res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]		; <i32> [#uses=2]
	%2 = load i32* %P, align 4		; <i32> [#uses=1]
	%3 = icmp sgt i32 %2, 36		; <i1> [#uses=1]
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...)* @f2() nounwind		; <i32> [#uses=0]
	ret i32 %res.0

bb3:		; preds = %bb1
	ret i32 %res.0
}

declare i32 @f1(...)

declare i32 @f2(...)

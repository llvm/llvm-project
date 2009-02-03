; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep {%cv = phi i32}
; RUN: llvm-as < %s | opt -gvn | llvm-dis | grep {%bv = phi i32}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

define i32 @g(i32* %b, i32* %c) nounwind {
entry:
	%g = alloca i32		; <i32*> [#uses=4]
	%t1 = icmp eq i32* %b, null		; <i1> [#uses=1]
	br i1 %t1, label %bb, label %bb1

bb:		; preds = %entry
	%t2 = load i32* %c, align 4		; <i32> [#uses=1]
	%t3 = add i32 %t2, 1		; <i32> [#uses=1]
	store i32 %t3, i32* %g, align 4
	br label %bb2

bb1:		; preds = %entry
	%t5 = load i32* %b, align 4		; <i32> [#uses=1]
	%t6 = add i32 %t5, 1		; <i32> [#uses=1]
	store i32 %t6, i32* %g, align 4
	br label %bb2

bb2:		; preds = %bb1, %bb
	%c_addr.0 = phi i32* [ %g, %bb1 ], [ %c, %bb ]		; <i32*> [#uses=1]
	%b_addr.0 = phi i32* [ %b, %bb1 ], [ %g, %bb ]		; <i32*> [#uses=1]
	%cv = load i32* %c_addr.0, align 4		; <i32> [#uses=1]
	%bv = load i32* %b_addr.0, align 4		; <i32> [#uses=1]
	%ret = add i32 %cv, %bv		; <i32> [#uses=1]
	ret i32 %ret
}


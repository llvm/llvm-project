; RUN: opt < %s -gvn | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"
@sort_value = external global [256 x i32], align 32		; <[256 x i32]*> [#uses=2]

define i32 @Quiesce(i32 %alpha, i32 %beta, i32 %wtm, i32 %ply) nounwind {
entry:
	br label %bb22

bb22:		; preds = %bb23, %bb22, %entry
	br i1 false, label %bb23, label %bb22

bb23:		; preds = %bb23, %bb22
	%sortv.233 = phi i32* [ getelementptr ([256 x i32]* @sort_value, i32 0, i32 0), %bb22 ], [ %sortv.2, %bb23 ]		; <i32*> [#uses=1]
	%0 = load i32* %sortv.233, align 4		; <i32> [#uses=0]
	%sortv.2 = getelementptr [256 x i32]* @sort_value, i32 0, i32 0		; <i32*> [#uses=1]
	br i1 false, label %bb23, label %bb22
}

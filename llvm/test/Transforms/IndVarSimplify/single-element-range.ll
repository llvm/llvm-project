; RUN: opt < %s -indvars

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv6-apple-darwin10"

define void @sqlite3_free_table(i8** %azResult) nounwind {
entry:
	br i1 undef, label %return, label %bb

bb:		; preds = %entry
	%0 = load i8*, i8** undef, align 4		; <i8*> [#uses=2]
	%1 = ptrtoint i8* %0 to i32		; <i32> [#uses=1]
	%2 = icmp sgt i8* %0, inttoptr (i32 1 to i8*)		; <i1> [#uses=1]
	br i1 %2, label %bb1, label %bb5

bb1:		; preds = %bb1, %bb
	%i.01 = phi i32 [ %3, %bb1 ], [ 1, %bb ]		; <i32> [#uses=1]
	%3 = add i32 %i.01, 1		; <i32> [#uses=2]
	%4 = icmp slt i32 %3, %1		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb5

bb5:		; preds = %bb1, %bb
	ret void

return:		; preds = %entry
	ret void
}

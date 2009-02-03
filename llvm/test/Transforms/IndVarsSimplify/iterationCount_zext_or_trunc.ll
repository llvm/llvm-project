; RUN: llvm-as < %s | opt -indvars -disable-output

; ModuleID = 'testcase.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i686-pc-linux-gnu"

define i32 @testcase(i5 zeroext  %k) {
entry:
	br label %bb2

bb:		; preds = %bb2
	%tmp1 = add i32 %tmp2, %result		; <i32> [#uses=1]
	%indvar_next1 = add i5 %k_0, 1		; <i5> [#uses=1]
	br label %bb2

bb2:		; preds = %bb, %entry
	%k_0 = phi i5 [ 0, %entry ], [ %indvar_next1, %bb ]		; <i5> [#uses=2]
	%result = phi i32 [ 0, %entry ], [ %tmp1, %bb ]		; <i32> [#uses=2]
	%tmp2 = zext i5 %k_0 to i32		; <i32> [#uses=1]
	%exitcond = icmp eq i32 %tmp2, 16		; <i1> [#uses=1]
	br i1 %exitcond, label %bb3, label %bb

bb3:		; preds = %bb2
	ret i32 %result
}

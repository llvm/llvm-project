; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution -scalar-evolution-max-iterations=0 | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-iterations=0 2>&1 | FileCheck %s
; PR1706

; CHECK: backedge-taken count is 13

define i32 @f() {
entry:
	br label %bb5

bb:		; preds = %bb5
	%tmp2 = shl i32 %j.0, 1		; <i32> [#uses=1]
	%tmp4 = add i32 %i.0, 268435456		; <i32> [#uses=1]
	br label %bb5

bb5:		; preds = %bb, %entry
	%j.0 = phi i32 [ 1, %entry ], [ %tmp2, %bb ]		; <i32> [#uses=2]
	%i.0 = phi i32 [ -1879048192, %entry ], [ %tmp4, %bb ]		; <i32> [#uses=2]
	%tmp7 = icmp slt i32 %i.0, 1610612736		; <i1> [#uses=1]
	br i1 %tmp7, label %bb, label %return

return:		; preds = %bb5
	ret i32 %j.0
}

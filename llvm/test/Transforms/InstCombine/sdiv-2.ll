; RUN: opt < %s -instcombine -disable-output
; PR3144

define fastcc i32 @func(i32 %length) nounwind {
entry:
	%0 = icmp ne i32 %length, -1		; <i1> [#uses=1]
	%iftmp.13.0 = select i1 %0, i128 0, i128 200000000		; <i128> [#uses=2]
	%1 = sdiv i128 %iftmp.13.0, 10		; <i128> [#uses=1]
	br label %bb5

bb5:		; preds = %bb8, %entry
	%v.0 = phi i128 [ 0, %entry ], [ %6, %bb8 ]		; <i128> [#uses=2]
	%2 = icmp sgt i128 %v.0, %1		; <i1> [#uses=1]
	br i1 %2, label %overflow, label %bb7

bb7:		; preds = %bb5
	%3 = mul i128 %v.0, 10		; <i128> [#uses=2]
	%4 = sub i128 %iftmp.13.0, 0		; <i128> [#uses=1]
	%5 = icmp slt i128 %4, %3		; <i1> [#uses=1]
	br i1 %5, label %overflow, label %bb8

bb8:		; preds = %bb7
	%6 = add i128 0, %3		; <i128> [#uses=1]
	br label %bb5

overflow:		; preds = %bb7, %bb5
	ret i32 1
}

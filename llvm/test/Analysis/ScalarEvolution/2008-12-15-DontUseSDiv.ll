; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 2>&1 | FileCheck %s

; CHECK: /u 5

define i8 @foo0(i8 %i0) nounwind {
entry:
	br label %bb1

bb:		; preds = %bb1
	%0 = add i8 %j.0, 1		; <i8> [#uses=1]
	%1 = add i8 %i.0, 5		; <i8> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%j.0 = phi i8 [ 0, %entry ], [ %0, %bb ]		; <i8> [#uses=2]
	%i.0 = phi i8 [ %i0, %entry ], [ %1, %bb ]		; <i8> [#uses=2]
	%2 = icmp sgt i8 %i.0, 100		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

return:		; preds = %bb1
	ret i8 %j.0
}

; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution 2>&1 |  FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 2>&1 |  FileCheck %s
; XFAIL: *

; CHECK: (((-1 * %i0) + (100005 smax %i0)) /u 5)

define i32 @foo0(i32 %i0) nounwind {
entry:
	br label %bb1

bb:		; preds = %bb1
	%0 = add i32 %j.0, 1		; <i32> [#uses=1]
	%1 = add i32 %i.0, 5		; <i32> [#uses=1]
	br label %bb1

bb1:		; preds = %bb, %entry
	%j.0 = phi i32 [ 0, %entry ], [ %0, %bb ]		; <i32> [#uses=2]
	%i.0 = phi i32 [ %i0, %entry ], [ %1, %bb ]		; <i32> [#uses=2]
	%2 = icmp sgt i32 %i.0, 100000		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb

return:		; preds = %bb1
	ret i32 %j.0
}

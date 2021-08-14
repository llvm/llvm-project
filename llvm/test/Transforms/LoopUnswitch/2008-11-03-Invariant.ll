; REQUIRES: asserts
; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -verify-memoryssa -stats -disable-output 2>&1 | FileCheck %s
; PR 3170

define i32 @a(i32 %x, i32 %y) nounwind {
; CHECK: 1 loop-unswitch - Number of branches unswitched
; CHECK-NOT: Number of branches unswitched

entry:
	%0 = icmp ult i32 0, %y		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %bb4

bb.nph:		; preds = %entry
	%1 = icmp eq i32 %x, 0		; <i1> [#uses=1]
	br label %bb

bb:		; preds = %bb.nph, %bb3
	%i.01 = phi i32 [ %3, %bb3 ], [ 0, %bb.nph ]		; <i32> [#uses=1]
	br i1 %1, label %bb2, label %bb1

bb1:		; preds = %bb
	%2 = tail call i32 (...) @b() nounwind		; <i32> [#uses=0]
	br label %bb2

bb2:		; preds = %bb, %bb1
	%3 = add i32 %i.01, 1		; <i32> [#uses=2]
	br label %bb3

bb3:		; preds = %bb2
	%i.0 = phi i32 [ %3, %bb2 ]		; <i32> [#uses=1]
	%4 = icmp ult i32 %i.0, %y		; <i1> [#uses=1]
	br i1 %4, label %bb, label %bb3.bb4_crit_edge

bb3.bb4_crit_edge:		; preds = %bb3
	br label %bb4

bb4:		; preds = %bb3.bb4_crit_edge, %entry
	ret i32 0
}

declare i32 @b(...)

; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution 2>&1 | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 2>&1 | FileCheck %s
; PR2070

; CHECK: Printing analysis 'Scalar Evolution Analysis' for function 'a'
; CHECK-NOT: smax

define i32 @a(i32 %x) nounwind  {
entry:
	icmp sgt i32 %x, 1		; <i1>:0 [#uses=1]
	br i1 %0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%z.02 = phi i32 [ %1, %bb1 ], [ 1, %bb.nph ]		; <i32> [#uses=1]
	%i.01 = phi i32 [ %2, %bb1 ], [ 1, %bb.nph ]		; <i32> [#uses=2]
	mul i32 %z.02, %i.01		; <i32>:1 [#uses=2]
	add i32 %i.01, 1		; <i32>:2 [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	icmp slt i32 %2, %x		; <i1>:3 [#uses=1]
	br i1 %3, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:		; preds = %bb1
	%.lcssa = phi i32 [ %1, %bb1 ]		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	%z.0.lcssa = phi i32 [ %.lcssa, %bb1.bb2_crit_edge ], [ 1, %entry ]		; <i32> [#uses=1]
	ret i32 %z.0.lcssa
}

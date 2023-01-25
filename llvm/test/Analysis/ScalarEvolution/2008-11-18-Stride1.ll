; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

; CHECK: Loop %bb: backedge-taken count is ((-5 + %x) /u 3)
; CHECK: Loop %bb: constant max backedge-taken count is 1431655764


; ScalarEvolution can't compute a trip count because it doesn't know if
; dividing by the stride will have a remainder. This could theoretically
; be teaching it how to use a more elaborate trip count computation.

define i32 @f(i32 %x) nounwind readnone {
entry:
	%0 = icmp ugt i32 %x, 4		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb.nph, %bb1
	%indvar = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb1 ]		; <i32> [#uses=2]
	%tmp = mul i32 %indvar, -3		; <i32> [#uses=1]
	%x_addr.04 = add i32 %tmp, %x		; <i32> [#uses=1]
	%1 = add i32 %x_addr.04, -3		; <i32> [#uses=2]
	br label %bb1

bb1:		; preds = %bb
	%2 = icmp ugt i32 %1, 4		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %2, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:		; preds = %bb1
	%.lcssa = phi i32 [ %1, %bb1 ]		; <i32> [#uses=1]
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	%x_addr.0.lcssa = phi i32 [ %.lcssa, %bb1.bb2_crit_edge ], [ %x, %entry ]		; <i32> [#uses=1]
	ret i32 %x_addr.0.lcssa
}

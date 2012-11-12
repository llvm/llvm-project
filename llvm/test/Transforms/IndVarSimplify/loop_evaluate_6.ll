; RUN: opt < %s -indvars -loop-deletion -S | grep phi | count 1
; XFAIL: *

; Indvars can't evaluate this loop, because ScalarEvolution can't compute
; an exact trip count, because it doesn't know if dividing by the stride will
; have a remainder. It could be done with more aggressive VRP though.

define i32 @test(i32 %x_offs) nounwind readnone {
entry:
	%0 = icmp sgt i32 %x_offs, 4		; <i1> [#uses=1]
	br i1 %0, label %bb.nph, label %bb2

bb.nph:		; preds = %entry
	br label %bb

bb:		; preds = %bb1, %bb.nph
	%x_offs_addr.01 = phi i32 [ %1, %bb1 ], [ %x_offs, %bb.nph ]		; <i32> [#uses=1]
	%1 = add i32 %x_offs_addr.01, -4		; <i32> [#uses=3]
	br label %bb1

bb1:		; preds = %bb
	%2 = icmp sgt i32 %1, 4		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb1.bb2_crit_edge

bb1.bb2_crit_edge:		; preds = %bb1
	br label %bb2

bb2:		; preds = %bb1.bb2_crit_edge, %entry
	%x_offs_addr.0.lcssa = phi i32 [ %1, %bb1.bb2_crit_edge ], [ %x_offs, %entry ]		; <i32> [#uses=1]
	ret i32 %x_offs_addr.0.lcssa
}

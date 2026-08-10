; RUN: llc < %s -mtriple=i686-- | grep align | count 4

; TODO: Is it a good idea to align inner loops? It's hard to know without
; knowing what their trip counts are, or other dynamic information. For
; now, CodeGen aligns all loops.

@x = external dso_local global ptr		; <ptr> [#uses=1]

define i32 @t(i32 %a, i32 %b) nounwind readonly ssp {
entry:
	%0 = icmp eq i32 %a, 0		; <i1> [#uses=1]
	br i1 %0, label %bb5, label %bb.nph12

bb.nph12:		; preds = %entry
	%1 = icmp eq i32 %b, 0		; <i1> [#uses=1]
	%2 = load ptr, ptr @x, align 8		; <ptr> [#uses=1]
	br i1 %1, label %bb2.preheader, label %bb2.preheader.us

bb2.preheader.us:		; preds = %bb2.bb3_crit_edge.us, %bb.nph12
	%indvar18 = phi i32 [ 0, %bb.nph12 ], [ %indvar.next19, %bb2.bb3_crit_edge.us ]		; <i32> [#uses=2]
	%sum.111.us = phi i32 [ 0, %bb.nph12 ], [ %4, %bb2.bb3_crit_edge.us ]		; <i32> [#uses=0]
	%tmp16 = mul i32 %indvar18, %a		; <i32> [#uses=1]
	br label %bb1.us

bb1.us:		; preds = %bb1.us, %bb2.preheader.us
	%indvar = phi i32 [ 0, %bb2.preheader.us ], [ %indvar.next, %bb1.us ]		; <i32> [#uses=2]
	%tmp17 = add i32 %indvar, %tmp16		; <i32> [#uses=1]
	%tmp. = zext i32 %tmp17 to i64		; <i64> [#uses=1]
	%3 = getelementptr i32, ptr %2, i64 %tmp.		; <ptr> [#uses=1]
	%4 = load i32, ptr %3, align 4		; <i32> [#uses=2]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %b		; <i1> [#uses=1]
	br i1 %exitcond, label %bb2.bb3_crit_edge.us, label %bb1.us

bb2.bb3_crit_edge.us:		; preds = %bb1.us
	%indvar.next19 = add i32 %indvar18, 1		; <i32> [#uses=2]
	%exitcond22 = icmp eq i32 %indvar.next19, %a		; <i1> [#uses=1]
	br i1 %exitcond22, label %bb5, label %bb2.preheader.us

bb2.preheader:		; preds = %bb2.preheader, %bb.nph12
	%indvar24 = phi i32 [ %indvar.next25, %bb2.preheader ], [ 0, %bb.nph12 ]		; <i32> [#uses=1]
	%indvar.next25 = add i32 %indvar24, 1		; <i32> [#uses=2]
	%exitcond28 = icmp eq i32 %indvar.next25, %a		; <i1> [#uses=1]
	br i1 %exitcond28, label %bb5, label %bb2.preheader

bb5:		; preds = %bb2.preheader, %bb2.bb3_crit_edge.us, %entry
	%sum.1.lcssa = phi i32 [ 0, %entry ], [ 0, %bb2.preheader ], [ %4, %bb2.bb3_crit_edge.us ]		; <i32> [#uses=1]
	ret i32 %sum.1.lcssa
}

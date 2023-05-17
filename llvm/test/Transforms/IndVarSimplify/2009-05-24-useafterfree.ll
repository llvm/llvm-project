; RUN: opt < %s -passes=indvars
; PR4258
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define void @0(ptr, ptr, i32, i32) nounwind {
	br i1 false, label %bb.nph1.preheader, label %.outer._crit_edge

bb.nph1.preheader:		; preds = %4
	%smax = select i1 false, i32 -1, i32 0		; <i32> [#uses=1]
	%tmp12 = sub i32 0, %smax		; <i32> [#uses=1]
	br label %bb.nph1

bb.nph1:		; preds = %.outer, %bb.nph1.preheader
	br i1 undef, label %bb.nph3.preheader, label %.outer

bb.nph3.preheader:		; preds = %bb.nph1
	br label %bb.nph3

bb.nph3:		; preds = %bb.nph3, %bb.nph3.preheader
	%indvar7 = phi i32 [ %indvar.next8, %bb.nph3 ], [ 0, %bb.nph3.preheader ]		; <i32> [#uses=3]
	%tmp9 = mul i32 %indvar7, -1		; <i32> [#uses=1]
	%tmp13 = add i32 %tmp9, %tmp12		; <i32> [#uses=1]
	%tmp14 = add i32 %tmp13, -2		; <i32> [#uses=1]
	%5 = icmp sgt i32 %tmp14, 0		; <i1> [#uses=1]
	%indvar.next8 = add i32 %indvar7, 1		; <i32> [#uses=1]
	br i1 %5, label %bb.nph3, label %.outer.loopexit

.outer.loopexit:		; preds = %bb.nph3
	%indvar7.lcssa = phi i32 [ %indvar7, %bb.nph3 ]		; <i32> [#uses=0]
	br label %.outer

.outer:		; preds = %.outer.loopexit, %bb.nph1
	br i1 undef, label %bb.nph1, label %.outer._crit_edge.loopexit

.outer._crit_edge.loopexit:		; preds = %.outer
	br label %.outer._crit_edge

.outer._crit_edge:		; preds = %.outer._crit_edge.loopexit, %4
	ret void
}

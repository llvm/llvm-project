; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

; CHECK: @f
; CHECK: Loop %bb16.preheader: backedge-taken count is (-1 + %c.idx.val)

define i32 @f(i32 %c.idx.val) {

bb2:
	%k.018 = add i32 %c.idx.val, -1		; <i32> [#uses=2]
	%a14 = icmp slt i32 %k.018, 0		; <i1> [#uses=1]
	br i1 %a14, label %bb19, label %bb16.preheader

bb16.preheader:
	%k.019 = phi i32 [ %k.0, %bb18 ], [ %k.018, %bb2 ]		; <i32> [#uses=5]
	%x = phi i32 [ 0, %bb2 ], [ %x.1, %bb18]
	br label %bb18

bb18:		; preds = %bb18.loopexit
	%x.1 = add i32 %x, 1
	%k.0 = add i32 %k.019, -1		; <i32> [#uses=2]
	%a107 = icmp slt i32 %k.0, 0		; <i1> [#uses=1]
	br i1 %a107, label %bb18.bb19_crit_edge, label %bb16.preheader

bb18.bb19_crit_edge:
	ret i32 %x

bb19:
	ret i32 0

}

; RUN: opt -S -licm -loop-unroll < %s
;
; This test contains a carefully rotated set of three nested loops. The middle
; loop can be unrolled leaving one copy of the inner loop inside the outer
; loop. Because of how LICM works, when this middle loop is unrolled and
; removed, its alias set tracker is destroyed and no longer available when LICM
; runs on the outer loop.

define void @f() {
entry:
  br label %l1

l2.l1.loopexit_crit_edge:
  br label %l1.loopexit

l1.loopexit:
  br label %l1.backedge

l1:
  br i1 undef, label %l1.backedge, label %l2.preheader

l1.backedge:
  br label %l1

l2.preheader:
  br i1 true, label %l1.loopexit, label %l3.preheader.lr.ph

l3.preheader.lr.ph:
  br label %l3.preheader

l2.loopexit:
  br i1 true, label %l2.l1.loopexit_crit_edge, label %l3.preheader

l3.preheader:
  br label %l3

l3:
  br i1 true, label %l3, label %l2.loopexit
}

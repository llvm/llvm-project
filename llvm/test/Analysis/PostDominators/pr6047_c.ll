; RUN: opt < %s -passes='print<postdomtree>' 2>&1 | FileCheck %s
define internal void @f(i1 %arg) {
entry:
  br i1 %arg, label %bb35, label %bb3.i

bb3.i:
  br label %bb3.i

bb:
  br label %bb35

bb.i:
  br label %bb35

_float32_unpack.exit:
  br label %bb35

bb.i5:
  br label %bb35

_float32_unpack.exit8:
  br label %bb35

bb32.preheader:
  br label %bb35

bb3:
  br label %bb35

bb3.split.us:
  br label %bb35

bb.i4.us:
  br label %bb35

bb7.i.us:
  br label %bb35

bb.i4.us.backedge:
  br label %bb35

bb1.i.us:
  br label %bb35

bb6.i.us:
  br label %bb35

bb4.i.us:
  br label %bb35

bb8.i.us:
  br label %bb35

bb3.i.loopexit.us:
  br label %bb35

bb.nph21:
  br label %bb35

bb4:
  br label %bb35

bb5:
  br label %bb35

bb14.preheader:
  br label %bb35

bb.nph18:
  br label %bb35

bb8.us.preheader:
  br label %bb35

bb8.preheader:
  br label %bb35

bb8.us:
  br label %bb35

bb8:
  br label %bb35

bb15.loopexit:
  br label %bb35

bb15.loopexit2:
  br label %bb35

bb15:
  br label %bb35

bb16:
  br label %bb35

bb17.loopexit.split:
  br label %bb35

bb.nph14:
  br label %bb35

bb19:
  br label %bb35

bb20:
  br label %bb35

bb29.preheader:
  br label %bb35

bb.nph:
  br label %bb35

bb23.us.preheader:
  br label %bb35

bb23.preheader:
  br label %bb35

bb23.us:
  br label %bb35

bb23:
  br label %bb35

bb30.loopexit:
  br label %bb35

bb30.loopexit1:
  br label %bb35

bb30:
  br label %bb35

bb31:
  br label %bb35

bb35.loopexit:
  br label %bb35

bb35.loopexit3:
  br label %bb35

bb35:
  ret void
}
; CHECK: Inorder PostDominator Tree:
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %bb35
; CHECK-NEXT:       [3] %bb
; CHECK-NEXT:       [3] %bb.i
; CHECK-NEXT:       [3] %_float32_unpack.exit
; CHECK-NEXT:       [3] %bb.i5
; CHECK-NEXT:       [3] %_float32_unpack.exit8
; CHECK-NEXT:       [3] %bb32.preheader
; CHECK-NEXT:       [3] %bb3
; CHECK-NEXT:       [3] %bb3.split.us
; CHECK-NEXT:       [3] %bb.i4.us
; CHECK-NEXT:       [3] %bb7.i.us
; CHECK-NEXT:       [3] %bb.i4.us.backedge
; CHECK-NEXT:       [3] %bb1.i.us
; CHECK-NEXT:       [3] %bb6.i.us
; CHECK-NEXT:       [3] %bb4.i.us
; CHECK-NEXT:       [3] %bb8.i.us
; CHECK-NEXT:       [3] %bb3.i.loopexit.us
; CHECK-NEXT:       [3] %bb.nph21
; CHECK-NEXT:       [3] %bb4
; CHECK-NEXT:       [3] %bb5
; CHECK-NEXT:       [3] %bb14.preheader
; CHECK-NEXT:       [3] %bb.nph18
; CHECK-NEXT:       [3] %bb8.us.preheader
; CHECK-NEXT:       [3] %bb8.preheader
; CHECK-NEXT:       [3] %bb8.us
; CHECK-NEXT:       [3] %bb8
; CHECK-NEXT:       [3] %bb15.loopexit
; CHECK-NEXT:       [3] %bb15.loopexit2
; CHECK-NEXT:       [3] %bb15
; CHECK-NEXT:       [3] %bb16
; CHECK-NEXT:       [3] %bb17.loopexit.split
; CHECK-NEXT:       [3] %bb.nph14
; CHECK-NEXT:       [3] %bb19
; CHECK-NEXT:       [3] %bb20
; CHECK-NEXT:       [3] %bb29.preheader
; CHECK-NEXT:       [3] %bb.nph
; CHECK-NEXT:       [3] %bb23.us.preheader
; CHECK-NEXT:       [3] %bb23.preheader
; CHECK-NEXT:       [3] %bb23.us
; CHECK-NEXT:       [3] %bb23
; CHECK-NEXT:       [3] %bb30.loopexit
; CHECK-NEXT:       [3] %bb30.loopexit1
; CHECK-NEXT:       [3] %bb30
; CHECK-NEXT:       [3] %bb31
; CHECK-NEXT:       [3] %bb35.loopexit
; CHECK-NEXT:       [3] %bb35.loopexit3
; CHECK-NEXT:     [2] %entry
; CHECK-NEXT:     [2] %bb3.i
; CHECK-NEXT: Roots: %bb35 %bb3.i

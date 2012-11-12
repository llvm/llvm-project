; RUN: opt < %s -postdomtree -analyze | FileCheck %s
define internal void @f() {
entry:
  br i1 undef, label %bb35, label %bb3.i

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
; CHECK: [3] %entry

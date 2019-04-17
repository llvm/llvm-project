; RUN: opt < %s -S -gvn-sink | FileCheck %s

; Because %tmp17 has flipped operands to its equivalents %tmp14 and %tmp7, we
; can't sink the zext as we'd need a shuffling PHI in between.
;
; Just sinking the zext isn't profitable, so ensure nothing is sunk.

; CHECK-LABEL: @hoge
; CHECK-NOT: bb18.gvnsink.split
define void @hoge() {
bb:
  br i1 undef, label %bb4, label %bb11

bb4:                                              ; preds = %bb3
  br i1 undef, label %bb6, label %bb8

bb6:                                              ; preds = %bb5
  %tmp = zext i16 undef to i64
  %tmp7 = add i64 %tmp, undef
  br label %bb18

bb8:                                              ; preds = %bb5
  %tmp9 = zext i16 undef to i64
  br label %bb18

bb11:                                             ; preds = %bb10
  br i1 undef, label %bb12, label %bb15

bb12:                                             ; preds = %bb11
  %tmp13 = zext i16 undef to i64
  %tmp14 = add i64 %tmp13, undef
  br label %bb18

bb15:                                             ; preds = %bb11
  %tmp16 = zext i16 undef to i64
  %tmp17 = add i64 undef, %tmp16
  br label %bb18

bb18:                                             ; preds = %bb15, %bb12, %bb8, %bb6
  %tmp19 = phi i64 [ %tmp7, %bb6 ], [ undef, %bb8 ], [ %tmp14, %bb12 ], [ %tmp17, %bb15 ]
  unreachable
}

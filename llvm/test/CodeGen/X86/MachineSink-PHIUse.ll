; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-appel-darwin -disable-cgp-branch-opts -stats 2>&1 | grep "machine-sink"

define fastcc void @t() nounwind ssp {
entry:
  br i1 undef, label %bb, label %bb4

bb:                                               ; preds = %entry
  br i1 undef, label %return, label %bb3

bb3:                                              ; preds = %bb
  unreachable

bb4:                                              ; preds = %entry
  br i1 undef, label %bb.nph, label %return

bb.nph:                                           ; preds = %bb4
  br label %bb5

bb5:                                              ; preds = %bb9, %bb.nph
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp12, %bb9 ] ; <i64> [#uses=1]
  %tmp12 = add i64 %indvar, 1                     ; <i64> [#uses=2]
  %tmp13 = trunc i64 %tmp12 to i32                ; <i32> [#uses=0]
  br i1 undef, label %bb9, label %bb6

bb6:                                              ; preds = %bb5
  br i1 undef, label %bb9, label %bb7

bb7:                                              ; preds = %bb6
  br i1 undef, label %bb9, label %bb8

bb8:                                              ; preds = %bb7
  unreachable

bb9:                                              ; preds = %bb7, %bb6, %bb5
  br i1 undef, label %bb5, label %return

return:                                           ; preds = %bb9, %bb4, %bb
  ret void
}

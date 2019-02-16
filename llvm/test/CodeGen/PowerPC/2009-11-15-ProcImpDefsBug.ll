; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu

define void @gcov_exit() nounwind {
entry:
  br i1 undef, label %return, label %bb.nph341

bb.nph341:                                        ; preds = %entry
  br label %bb25

bb25:                                             ; preds = %read_fatal, %bb.nph341
  br i1 undef, label %bb49.1, label %bb48

bb48:                                             ; preds = %bb25
  br label %bb49.1

bb51:                                             ; preds = %bb48.4, %bb49.3
  switch i32 undef, label %bb58 [
    i32 0, label %rewrite
    i32 1734567009, label %bb59
  ]

bb58:                                             ; preds = %bb51
  br label %read_fatal

bb59:                                             ; preds = %bb51
  br i1 undef, label %bb60, label %bb3.i156

bb3.i156:                                         ; preds = %bb59
  br label %read_fatal

bb60:                                             ; preds = %bb59
  br i1 undef, label %bb78.preheader, label %rewrite

bb78.preheader:                                   ; preds = %bb60
  br i1 undef, label %bb62, label %bb80

bb62:                                             ; preds = %bb78.preheader
  br i1 undef, label %bb64, label %read_mismatch

bb64:                                             ; preds = %bb62
  br i1 undef, label %bb65, label %read_mismatch

bb65:                                             ; preds = %bb64
  br i1 undef, label %bb75, label %read_mismatch

read_mismatch:                                    ; preds = %bb98, %bb119.preheader, %bb72, %bb71, %bb65, %bb64, %bb62
  br label %read_fatal

bb71:                                             ; preds = %bb75
  br i1 undef, label %bb72, label %read_mismatch

bb72:                                             ; preds = %bb71
  br i1 undef, label %bb73, label %read_mismatch

bb73:                                             ; preds = %bb72
  unreachable

bb74:                                             ; preds = %bb75
  br label %bb75

bb75:                                             ; preds = %bb74, %bb65
  br i1 undef, label %bb74, label %bb71

bb80:                                             ; preds = %bb78.preheader
  unreachable

read_fatal:                                       ; preds = %read_mismatch, %bb3.i156, %bb58
  br i1 undef, label %return, label %bb25

rewrite:                                          ; preds = %bb60, %bb51
  br i1 undef, label %bb94, label %bb119.preheader

bb94:                                             ; preds = %rewrite
  unreachable

bb119.preheader:                                  ; preds = %rewrite
  br i1 undef, label %read_mismatch, label %bb98

bb98:                                             ; preds = %bb119.preheader
  br label %read_mismatch

return:                                           ; preds = %read_fatal, %entry
  ret void

bb49.1:                                           ; preds = %bb48, %bb25
  br i1 undef, label %bb49.2, label %bb48.2

bb49.2:                                           ; preds = %bb48.2, %bb49.1
  br i1 undef, label %bb49.3, label %bb48.3

bb48.2:                                           ; preds = %bb49.1
  br label %bb49.2

bb49.3:                                           ; preds = %bb48.3, %bb49.2
  %c_ix.0.3 = phi i32 [ undef, %bb48.3 ], [ undef, %bb49.2 ] ; <i32> [#uses=1]
  br i1 undef, label %bb51, label %bb48.4

bb48.3:                                           ; preds = %bb49.2
  store i64* undef, i64** undef, align 4
  br label %bb49.3

bb48.4:                                           ; preds = %bb49.3
  %0 = getelementptr inbounds [5 x i64*], [5 x i64*]* undef, i32 0, i32 %c_ix.0.3 ; <i64**> [#uses=0]
  br label %bb51
}

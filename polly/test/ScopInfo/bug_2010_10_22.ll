; RUN: opt %loadPolly %defaultOpts -polly-analyze-ir %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define fastcc void @_Z8wavModelR5Mixer() {
entry:
  br label %bb230

bb230:                                            ; preds = %bb233, %bb.nph433
  %indvar600 = phi i64 [ 0, %entry ], [ %tmp610, %bb233 ]
  %tmp217 = add i64 %indvar600, -1
  %tmp204 = trunc i64 %tmp217 to i32
  %tmp205 = zext i32 %tmp204 to i64
  %tmp206 = add i64 %tmp205, 1
  %tmp610 = add i64 %indvar600, 1
  br i1 false, label %bb231.preheader, label %bb233

bb231.preheader:                                  ; preds = %bb230
  br label %bb231

bb231:                                            ; preds = %bb231, %bb231.preheader
  %indvar589 = phi i64 [ %tmp611, %bb231 ], [ 0, %bb231.preheader ]
  %tmp611 = add i64 %indvar589, 1
  %exitcond207 = icmp eq i64 %tmp611, %tmp206
  br i1 %exitcond207, label %bb233.loopexit, label %bb231

bb233.loopexit:                                   ; preds = %bb231
  br label %bb233

bb233:                                            ; preds = %bb233.loopexit, %bb230
  %exitcond213 = icmp eq i64 %tmp610, 0
  br i1 %exitcond213, label %bb241, label %bb230

bb241:                                            ; preds = %bb233, %bb228
  br label %bb244.preheader

bb244.preheader:                                  ; preds = %bb241, %bb176
  br i1 undef, label %bb245, label %bb.nph416

bb.nph416:                                        ; preds = %bb244.preheader
  unreachable

bb245:                                            ; preds = %bb244.preheader
  ret void
}

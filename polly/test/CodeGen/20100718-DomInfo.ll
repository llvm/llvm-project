; RUN: opt %loadNPMPolly -passes=polly-codegen -verify-dom-info -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @intrapred_luma_16x16(i32 %predmode) nounwind {
entry:
  switch i32 %predmode, label %bb81 [
    i32 0, label %bb25
    i32 1, label %bb26
  ]

bb23:                                             ; preds = %bb25
  %indvar.next95 = add i64 %indvar94, 1           ; <i64> [#uses=1]
  br label %bb25

bb25:                                             ; preds = %bb23, %entry
  %indvar94 = phi i64 [ %indvar.next95, %bb23 ], [ 0, %entry ] ; <i64> [#uses=1]
  br i1 false, label %bb23, label %return

bb26:                                             ; preds = %entry
  ret void

bb81:                                             ; preds = %entry
  ret void

return:                                           ; preds = %bb25
  ret void
}

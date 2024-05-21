; RUN: opt %loadNPMPolly -passes=polly-codegen -verify-dom-info -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @getNonAffNeighbour() nounwind {
entry:
  br i1 undef, label %bb, label %bb6

bb:                                               ; preds = %entry
  br i1 false, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  br label %bb16

bb2:                                              ; preds = %bb
  br i1 false, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  br label %bb16

bb4:                                              ; preds = %bb2
  br label %bb16

bb6:                                              ; preds = %entry
  br i1 false, label %bb7, label %bb9

bb7:                                              ; preds = %bb6
  br label %bb16

bb9:                                              ; preds = %bb6
  br label %bb16

bb16:                                             ; preds = %bb9, %bb7, %bb4, %bb3, %bb1
  ret void
}

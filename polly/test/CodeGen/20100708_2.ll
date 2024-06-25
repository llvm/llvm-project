; RUN: opt %loadNPMPolly -passes=polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @init_array() nounwind {
bb:
  br label %bb1

bb1:                                              ; preds = %bb4, %bb
  br i1 undef, label %bb2, label %bb5

bb2:                                              ; preds = %bb3, %bb1
  %indvar = phi i64 [ %indvar.next, %bb3 ], [ 0, %bb1 ] ; <i64> [#uses=1]
  %tmp3 = trunc i64 undef to i32                  ; <i32> [#uses=1]
  br i1 false, label %bb3, label %bb4

bb3:                                              ; preds = %bb2
  %tmp = srem i32 %tmp3, 1024                     ; <i32> [#uses=0]
  store double undef, ptr undef
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb2

bb4:                                              ; preds = %bb2
  br label %bb1

bb5:                                              ; preds = %bb1
  ret void
}

; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @clause_SetSplitField(i32 %Length) nounwind inlinehint {
entry:
  br i1 undef, label %bb1, label %bb6

bb1:                                              ; preds = %entry
  unreachable

bb6:                                              ; preds = %entry
  %tmp = zext i32 %Length to i64                  ; <i64> [#uses=1]
  br label %bb8

bb7:                                              ; preds = %bb8
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=1]
  br label %bb8

bb8:                                              ; preds = %bb7, %bb6
  %indvar = phi i64 [ %indvar.next, %bb7 ], [ 0, %bb6 ] ; <i64> [#uses=2]
  %exitcond = icmp ne i64 %indvar, %tmp           ; <i1> [#uses=1]
  br i1 %exitcond, label %bb7, label %return

return:                                           ; preds = %bb8
  ret void
}

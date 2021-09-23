; RUN: opt %loadPolly -polly-codegen < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define hidden void @luaD_callhook() nounwind {
entry:
  br i1 undef, label %bb, label %return

bb:                                               ; preds = %entry
  br i1 undef, label %bb1, label %return

bb1:                                              ; preds = %bb
  %0 = sub nsw i64 undef, undef                   ; <i64> [#uses=1]
  br i1 false, label %bb2, label %bb3

bb2:                                              ; preds = %bb1
  br label %bb4

bb3:                                              ; preds = %bb1
  br label %bb4

bb4:                                              ; preds = %bb3, %bb2
  br i1 undef, label %bb5, label %bb6

bb5:                                              ; preds = %bb4
  unreachable

bb6:                                              ; preds = %bb4
  %1 = getelementptr inbounds i8, i8* undef, i64 %0   ; <i8*> [#uses=0]
  ret void

return:                                           ; preds = %bb, %entry
  ret void
}

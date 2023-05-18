; RUN: opt %loadPolly  -polly-codegen -disable-output < %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define void @matrixTranspose(ptr %A) nounwind {
entry:
  br label %bb4

bb:                                               ; preds = %bb4
  %0 = add nsw i32 %i.0, 1                        ; <i32> [#uses=1]
  br label %bb2

bb1:                                              ; preds = %bb2
  %1 = getelementptr inbounds ptr, ptr %A, i64 0  ; <ptr> [#uses=0]
  %2 = getelementptr inbounds ptr, ptr %A, i64 0  ; <ptr> [#uses=0]
  %3 = getelementptr inbounds ptr, ptr %A, i64 0  ; <ptr> [#uses=0]
  %4 = sext i32 %j.0 to i64                       ; <i64> [#uses=1]
  %5 = getelementptr inbounds ptr, ptr %A, i64 %4 ; <ptr> [#uses=1]
  %6 = load ptr, ptr %5, align 8                  ; <ptr> [#uses=0]
  %7 = add nsw i32 %j.0, 1                        ; <i32> [#uses=1]
  br label %bb2

bb2:                                              ; preds = %bb1, %bb
  %j.0 = phi i32 [ %0, %bb ], [ %7, %bb1 ]        ; <i32> [#uses=3]
  %8 = icmp sle i32 %j.0, 50                      ; <i1> [#uses=1]
  br i1 %8, label %bb1, label %bb3

bb3:                                              ; preds = %bb2
  %9 = add nsw i32 %i.0, 1                        ; <i32> [#uses=1]
  br label %bb4

bb4:                                              ; preds = %bb3, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %9, %bb3 ]      ; <i32> [#uses=3]
  %10 = icmp sle i32 %i.0, 50                     ; <i1> [#uses=1]
  br i1 %10, label %bb, label %return

return:                                           ; preds = %bb4
  ret void
}

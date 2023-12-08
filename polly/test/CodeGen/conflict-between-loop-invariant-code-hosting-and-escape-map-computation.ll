; RUN: opt %loadPolly -polly-process-unprofitable -polly-codegen -disable-output < %s
;
; CHECK: store i32 %tmp14_p_scalar_, ptr %tmp14.s2a
; CHECK: %tmp14.final_reload = load i32, ptr %tmp14.s2a
; CHECK: %tmp17b.final_reload = load i32, ptr %tmp17b.preload.s2a
; CHECK: %tmp17.final_reload = load i32, ptr %tmp17.preload.s2a
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @hoge(ptr %arg, i32 %arg4) #0 {
bb:
  br label %bb5

bb5:                                              ; preds = %bb
  br i1 undef, label %bb6, label %bb18

bb6:                                              ; preds = %bb5
  %tmp8 = getelementptr inbounds i8, ptr %arg, i64 4
  %tmp9 = getelementptr inbounds i8, ptr %tmp8, i64 20
  br label %bb10

bb10:                                             ; preds = %bb10, %bb6
  %tmp11 = phi i32 [ %tmp12, %bb10 ], [ 2, %bb6 ]
  %tmp12 = add nuw nsw i32 %tmp11, 1
  br i1 false, label %bb10, label %bb13

bb13:                                             ; preds = %bb10
  %tmp14 = load i32, ptr %tmp9, align 4
  %tmp15 = getelementptr inbounds i8, ptr %tmp9, i64 4
  %tmp17 = load i32, ptr %tmp15, align 4
  store i32 %tmp17, ptr %tmp9, align 4
  %tmp15b = getelementptr inbounds i8, ptr %tmp9, i64 8
  %tmp17b = load i32, ptr %tmp15b, align 4
  store i32 %tmp17b, ptr %tmp9, align 4
  br label %bb19

bb18:                                             ; preds = %bb5
  br label %bb19

bb19:                                             ; preds = %bb18, %bb13
  %tmp20 = phi i32 [ %tmp14, %bb13 ], [ %arg4, %bb18 ]
  %tmp21 = phi i32 [ %tmp17, %bb13 ], [ %arg4, %bb18 ]
  %tmp22 = phi i32 [ %tmp17b, %bb13 ], [ %arg4, %bb18 ]
  unreachable
}

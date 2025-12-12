; RUN: opt %loadNPMPolly -S -polly-vectorizer=stripmine '-passes=polly-custom<opt-isl>' -polly-debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

define void @ham(ptr %arg, ptr %arg1, i32 %arg2, i32 %arg3, ptr %arg4, i32 %arg5, i32 %arg6) {
bb:
  %getelementptr = getelementptr [7 x float], ptr null, i32 0, i32 %arg3
  br label %bb7

bb7:                                              ; preds = %bb11, %bb
  %phi = phi i32 [ 0, %bb ], [ %add16, %bb11 ]
  br label %bb8

bb8:                                              ; preds = %bb8, %bb7
  %phi9 = phi i32 [ 0, %bb7 ], [ %add, %bb8 ]
  %getelementptr10 = getelementptr [7 x float], ptr null, i32 0, i32 %phi9
  store float 0.000000e+00, ptr %getelementptr10, align 4
  %add = add i32 %phi9, 1
  %icmp = icmp eq i32 %phi9, 0
  br i1 %icmp, label %bb8, label %bb11

bb11:                                             ; preds = %bb8
  %load = load float, ptr %getelementptr, align 4
  store float %load, ptr %arg4, align 4
  %getelementptr12 = getelementptr [7 x float], ptr null, i32 0, i32 %arg5
  %load13 = load float, ptr %getelementptr12, align 4
  store float %load13, ptr %arg, align 4
  %getelementptr14 = getelementptr [7 x float], ptr null, i32 0, i32 %arg6
  %load15 = load float, ptr %getelementptr14, align 4
  store float %load15, ptr %arg1, align 4
  %add16 = add i32 %phi, 1
  %icmp17 = icmp ne i32 %phi, %arg2
  br i1 %icmp17, label %bb7, label %bb18

bb18:                                             ; preds = %bb11
  ret void
}
; CHECK:Schedule optimizer calculation exceeds ISL quota

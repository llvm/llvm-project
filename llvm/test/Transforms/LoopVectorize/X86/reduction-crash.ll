; RUN: opt -S -aa-pipeline= -passes=loop-vectorize -mcpu=prescott < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"
target triple = "i386-apple-darwin"

; PR15344
define void @test1(ptr nocapture %arg, i32 %arg1) nounwind {
; CHECK-LABEL: @test1(
; CHECK: preheader
; CHECK: insertelement <2 x double> zeroinitializer, double %tmp, i32 0
; CHECK: vector.memcheck

bb:
  br label %bb2

bb2:                                              ; preds = %bb
  %tmp = load double, ptr null, align 8
  br i1 undef, label %bb3, label %bb12

bb3:                                              ; preds = %bb3, %bb2
  %tmp4 = phi double [ %tmp9, %bb3 ], [ %tmp, %bb2 ]
  %tmp5 = phi i32 [ %tmp8, %bb3 ], [ 0, %bb2 ]
  %tmp6 = getelementptr inbounds [16 x double], ptr undef, i32 0, i32 %tmp5
  %tmp7 = load double, ptr %tmp6, align 4
  %tmp8 = add nsw i32 %tmp5, 1
  %tmp9 = fadd fast double %tmp4, undef
  %tmp10 = getelementptr inbounds float, ptr %arg, i32 %tmp5
  store float undef, ptr %tmp10, align 4
  %tmp11 = icmp eq i32 %tmp8, %arg1
  br i1 %tmp11, label %bb12, label %bb3

bb12:                                             ; preds = %bb3, %bb2
  %tmp13 = phi double [ %tmp, %bb2 ], [ %tmp9, %bb3 ]
  ret void
}

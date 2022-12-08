; RUN: opt < %s -S -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -passes=loop-unroll | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;; Check that we do emit expensive instructions to compute trip
;; counts when unrolling loops on the a2 (because we unroll a lot).

define i32 @test(i64 %v12, i8* %array, i64* %loc) {
; CHECK-LABEL: @test(
; CHECK: udiv
entry:
  %step = load i64, i64* %loc, !range !0
  br label %loop

loop:                                           ; preds = %entry, %loop
  %k.015 = phi i64 [ %v15, %loop ], [ %v12, %entry ]
  %v14 = getelementptr inbounds i8, i8* %array, i64 %k.015
  store i8 0, i8* %v14
  %v15 = add nuw nsw i64 %k.015, %step
  %v16 = icmp slt i64 %v15, 8193
  br i1 %v16, label %loop, label %loopexit

loopexit:                             ; preds = %loop
  ret i32 0
}

!0 = !{i64 1, i64 100}

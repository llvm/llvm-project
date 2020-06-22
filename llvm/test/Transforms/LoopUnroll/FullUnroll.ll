; RUN: opt -passes='default<O1>' -disable-verify --mtriple x86_64-pc-linux-gnu -new-pm-disable-loop-unrolling=true \
; RUN: -S -o - %s | FileCheck %s

; This checks that the loop full unroller will fire in the new pass manager
; when forced via #pragma in the source (or annotation in the code).
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; We don't end up deleting the loop, merely turning it infinite, but we remove
; everything inside of it so check for the loop structure and absence of
; conditional branches.
; CHECK-LABEL: bb
; CHECK: br label
; CHECK-NOT: br i1
; CHECK: br label
; CHECK-NOT: br i1

; Function Attrs: noinline nounwind optnone uwtable
define void @foo() #0 {
bb:
  %tmp = alloca [5 x i32*], align 16
  %tmp1 = alloca i32, align 4
  %tmp2 = alloca i32, align 4
  store i32 5, i32* %tmp1, align 4
  br label %bb3

bb3:                                              ; preds = %bb23, %bb
  %tmp4 = load i32, i32* %tmp1, align 4
  %tmp5 = icmp ne i32 %tmp4, 0
  br i1 %tmp5, label %bb6, label %bb24

bb6:                                              ; preds = %bb3
  store i32 0, i32* %tmp2, align 4
  br label %bb7

bb7:                                              ; preds = %bb20, %bb6
  %tmp8 = load i32, i32* %tmp2, align 4
  %tmp9 = icmp slt i32 %tmp8, 5
  br i1 %tmp9, label %bb10, label %bb23

bb10:                                             ; preds = %bb7
  %tmp11 = load i32, i32* %tmp2, align 4
  %tmp12 = sext i32 %tmp11 to i64
  %tmp13 = getelementptr inbounds [5 x i32*], [5 x i32*]* %tmp, i64 0, i64 %tmp12
  %tmp14 = load i32*, i32** %tmp13, align 8
  %tmp15 = icmp ne i32* %tmp14, null
  br i1 %tmp15, label %bb16, label %bb19

bb16:                                             ; preds = %bb10
  %tmp17 = load i32, i32* %tmp1, align 4
  %tmp18 = add nsw i32 %tmp17, -1
  store i32 %tmp18, i32* %tmp1, align 4
  br label %bb19

bb19:                                             ; preds = %bb16, %bb10
  br label %bb20

bb20:                                             ; preds = %bb19
  %tmp21 = load i32, i32* %tmp2, align 4
  %tmp22 = add nsw i32 %tmp21, 1
  store i32 %tmp22, i32* %tmp2, align 4
  br label %bb7, !llvm.loop !1

bb23:                                             ; preds = %bb7
  br label %bb3

bb24:                                             ; preds = %bb3
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.unroll.full"}

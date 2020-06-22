; RUN: opt -loop-unroll-and-jam -allow-unroll-and-jam -verify-loop-info < %s -S | FileCheck %s
; RUN: opt -passes='unroll-and-jam,verify<loops>' -allow-unroll-and-jam < %s -S | FileCheck %s

; Check that the newly created loops to not fail to be added to LI
; This test deliberately disables UnJ on the middle loop, performing it instead on the
; outer of 3 nested loops. The (new) inner loops need to be added to LI.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define i32 @test() {
; CHECK-LABEL: test
; CHECK:       for.cond17.preheader:
; CHECK:    br label %for.cond20.preheader
; CHECK:       for.cond20.preheader:
; CHECK:    br label %for.cond23.preheader
; CHECK:       for.cond23.preheader:
; CHECK:    br label %for.body25
; CHECK:       for.body25:
; CHECK:    br i1 [[CMP24:%.*]], label %for.body25, label %for.inc45
; CHECK:       for.inc45:
; CHECK:    br label %for.body25.1
; CHECK:       for.inc48:
; CHECK:    br i1 [[CMP18_3:%.*]], label %for.cond20.preheader, label %for.end50
; CHECK:       for.end50:
; CHECK:    ret i32 0
; CHECK:       for.body25.1:
; CHECK:    br i1 [[CMP24_1:%.*]], label %for.body25.1, label %for.inc45.1
; CHECK:       for.inc45.1:
; CHECK:    br label %for.body25.2
; CHECK:       for.body25.2:
; CHECK:    br i1 [[CMP24_2:%.*]], label %for.body25.2, label %for.inc45.2
; CHECK:       for.inc45.2:
; CHECK:    br label %for.body25.3
; CHECK:       for.body25.3:
; CHECK:    br i1 [[CMP24_3:%.*]], label %for.body25.3, label %for.inc45.3
; CHECK:       for.inc45.3:
; CHECK:    br i1 [[CMP21_3:%.*]], label %for.cond23.preheader, label %for.inc48
;
entry:
  %A = alloca [8 x [8 x i32]], align 16
  %B = alloca [8 x [8 x i32]], align 16
  %C = alloca [8 x [8 x i32]], align 16
  br label %for.cond17.preheader

for.cond17.preheader:                             ; preds = %for.inc14
  br label %for.cond20.preheader

for.cond20.preheader:                             ; preds = %for.cond17.preheader, %for.inc48
  %i.13 = phi i32 [ 0, %for.cond17.preheader ], [ %inc49, %for.inc48 ]
  br label %for.cond23.preheader

for.cond23.preheader:                             ; preds = %for.cond20.preheader, %for.inc45
  %j.12 = phi i32 [ 0, %for.cond20.preheader ], [ %inc46, %for.inc45 ]
  br label %for.body25

for.body25:                                       ; preds = %for.cond23.preheader, %for.body25
  %k.01 = phi i32 [ 0, %for.cond23.preheader ], [ %inc43, %for.body25 ]
  %idxprom26 = zext i32 %i.13 to i64
  %idxprom28 = zext i32 %j.12 to i64
  %arrayidx29 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %C, i64 0, i64 %idxprom26, i64 %idxprom28
  %0 = load i32, i32* %arrayidx29, align 4
  %idxprom30 = zext i32 %i.13 to i64
  %idxprom32 = zext i32 %k.01 to i64
  %arrayidx33 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %A, i64 0, i64 %idxprom30, i64 %idxprom32
  %1 = load i32, i32* %arrayidx33, align 4
  %idxprom34 = zext i32 %k.01 to i64
  %idxprom36 = zext i32 %j.12 to i64
  %arrayidx37 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %B, i64 0, i64 %idxprom34, i64 %idxprom36
  %2 = load i32, i32* %arrayidx37, align 4
  %mul = mul nsw i32 %1, %2
  %add = add nsw i32 %0, %mul
  %idxprom38 = zext i32 %i.13 to i64
  %idxprom40 = zext i32 %j.12 to i64
  %arrayidx41 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %C, i64 0, i64 %idxprom38, i64 %idxprom40
  store i32 %add, i32* %arrayidx41, align 4
  %inc43 = add nuw nsw i32 %k.01, 1
  %cmp24 = icmp ult i32 %k.01, 7
  br i1 %cmp24, label %for.body25, label %for.inc45

for.inc45:                                        ; preds = %for.body25
  %inc46 = add nuw nsw i32 %j.12, 1
  %cmp21 = icmp ult i32 %j.12, 7
  br i1 %cmp21, label %for.cond23.preheader, label %for.inc48, !llvm.loop !7

for.inc48:                                        ; preds = %for.inc45
  %inc49 = add nuw nsw i32 %i.13, 1
  %cmp18 = icmp ult i32 %i.13, 7
  br i1 %cmp18, label %for.cond20.preheader, label %for.end50, !llvm.loop !5

for.end50:                                        ; preds = %for.inc48
  ret i32 0
}

!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll_and_jam.count", i32 4}
!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.unroll_and_jam.disable"}

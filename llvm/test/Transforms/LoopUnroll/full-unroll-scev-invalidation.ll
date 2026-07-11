; RUN: opt -S -passes='loop(loop-idiom,indvars,loop-deletion,loop-unroll-full,loop-idiom,indvars),simplifycfg,instcombine' %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@g0 = global i64 0, align 8
@g15 = global i64 0, align 8
@g14 = global i64 0, align 8
@g5 = global i64 0, align 8
@f4_c7 = global i8 0, align 1
@g8 = global i64 0, align 8
@f4_c15 = global i8 0, align 1
@f4_c8 = global i8 0, align 1
@__chk = global i64 0, align 8

define void @f4() {
; CHECK-LABEL: define void @f4(
; CHECK-NOT: ret void
; CHECK: store i64 5, ptr @__chk
entry:
  %0 = load i64, ptr @g0, align 8
  %cmp.not = icmp eq i64 %0, 908375363948206739
  br i1 %cmp.not, label %if.end, label %if.then

if.then:
  store i64 1, ptr @g15, align 8
  br label %if.end

if.end:
  %1 = load i64, ptr @g15, align 8
  store i64 %1, ptr @g14, align 8
  store i64 %1, ptr @g0, align 8
  br label %lbl_b5

lbl_b5:
  %bb13.0 = phi i32 [ 0, %if.end ], [ %bb13.1.lcssa, %if.then8 ]
  %ov6.0 = phi i1 [ false, %if.end ], [ true, %if.then8 ]
  br label %lbl_b10

lbl_b10:
  %2 = phi i8 [ 0, %lbl_b5 ], [ 1, %lbl_b10 ]
  %3 = phi i64 [ %1, %lbl_b5 ], [ 0, %lbl_b10 ]
  %bb13.1 = phi i32 [ %bb13.0, %lbl_b5 ], [ %5, %lbl_b10 ]
  %conv1 = trunc i64 %3 to i32
  %4 = tail call range(i32 0, 33) i32 @llvm.ctpop.i32(i32 %conv1)
  %5 = and i32 %4, 1
  %loadedv = trunc nuw i8 %2 to i1
  br i1 %loadedv, label %if.then3, label %lbl_b10

if.then3:
  %.lcssa38 = phi i8 [ %2, %lbl_b10 ]
  %.lcssa36 = phi i64 [ %3, %lbl_b10 ]
  %bb13.1.lcssa = phi i32 [ %bb13.1, %lbl_b10 ]
  %.lcssa = phi i32 [ %5, %lbl_b10 ]
  %tobool4.not = icmp eq i32 %bb13.1.lcssa, 0
  br i1 %tobool4.not, label %cleanup.loopexit, label %if.then8

if.then8:
  br i1 %ov6.0, label %if.then10, label %lbl_b5

if.then10:
  %.lcssa38.lcssa39 = phi i8 [ %.lcssa38, %if.then8 ]
  %.lcssa36.lcssa37 = phi i64 [ %.lcssa36, %if.then8 ]
  %bb13.1.lcssa.lcssa35 = phi i32 [ %bb13.1.lcssa, %if.then8 ]
  %.lcssa.lcssa34 = phi i32 [ %.lcssa, %if.then8 ]
  %conv2.le.le = zext nneg i32 %.lcssa.lcssa34 to i64
  %storedv5.le = trunc nuw i32 %bb13.1.lcssa.lcssa35 to i8
  store i64 %.lcssa36.lcssa37, ptr @g5, align 8
  store i8 %.lcssa38.lcssa39, ptr @f4_c7, align 1
  store i64 %conv2.le.le, ptr @g8, align 8
  store i8 1, ptr @f4_c15, align 1
  store i8 %storedv5.le, ptr @f4_c8, align 1
  store i64 5, ptr @__chk, align 8
  br label %cleanup

cleanup.loopexit:
  %ov6.0.lcssa = phi i1 [ %ov6.0, %if.then3 ]
  %.lcssa38.lcssa = phi i8 [ %.lcssa38, %if.then3 ]
  %.lcssa36.lcssa = phi i64 [ %.lcssa36, %if.then3 ]
  %.lcssa.lcssa = phi i32 [ %.lcssa, %if.then3 ]
  %conv2.le.le32 = zext nneg i32 %.lcssa.lcssa to i64
  %storedv.le30 = zext i1 %ov6.0.lcssa to i8
  store i64 %.lcssa36.lcssa, ptr @g5, align 8
  store i8 %.lcssa38.lcssa, ptr @f4_c7, align 1
  store i64 %conv2.le.le32, ptr @g8, align 8
  store i8 %storedv.le30, ptr @f4_c15, align 1
  store i8 0, ptr @f4_c8, align 1
  br label %cleanup

cleanup:
  ret void
}

declare i32 @llvm.ctpop.i32(i32)

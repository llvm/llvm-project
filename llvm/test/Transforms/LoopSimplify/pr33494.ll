; RUN: opt -loop-unroll -loop-simplify -S  < %s | FileCheck %s

; This test is one of the tests of PR33494. Its compilation takes
; excessive time if we don't mark visited nodes while looking for
; constants in SCEV expression trees.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test_01(i32* nocapture %a) local_unnamed_addr {

; CHECK-LABEL: @test_01(

entry:
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 96
  %arrayidx.promoted51 = load i32, i32* %arrayidx, align 1
  br label %while.body

while.body:                                       ; preds = %entry, %while.end29
  %0 = phi i32 [ %arrayidx.promoted51, %entry ], [ %7, %while.end29 ]
  %cmp46 = icmp eq i32 %0, 1
  %conv47 = zext i1 %cmp46 to i32
  %1 = add i32 %0, 1
  %2 = icmp ult i32 %1, 3
  %div48 = select i1 %2, i32 %0, i32 0
  %cmp349 = icmp sgt i32 %div48, %conv47
  br i1 %cmp349, label %while.body4.lr.ph, label %while.end29

while.body4.lr.ph:                                ; preds = %while.body
  br label %while.body4

while.body4:                                      ; preds = %while.body4.lr.ph, %while.end28
  %3 = phi i32 [ %0, %while.body4.lr.ph ], [ %mul17.lcssa, %while.end28 ]
  br label %while.body13

while.body13:                                     ; preds = %while.body4, %while.end.split
  %mul1745 = phi i32 [ %3, %while.body4 ], [ %mul17, %while.end.split ]
  %4 = phi i32 [ 15872, %while.body4 ], [ %add, %while.end.split ]
  %mul = mul nsw i32 %mul1745, %mul1745
  %mul17 = mul nsw i32 %mul, %mul1745
  %cmp22 = icmp eq i32 %4, %mul17
  br i1 %cmp22, label %while.body13.split, label %while.end.split

while.body13.split:                               ; preds = %while.body13
  br label %while.cond19

while.cond19:                                     ; preds = %while.cond19, %while.body13.split
  br label %while.cond19

while.end.split:                                  ; preds = %while.body13
  %add = shl nsw i32 %4, 1
  %tobool12 = icmp eq i32 %add, 0
  br i1 %tobool12, label %while.end28, label %while.body13

while.end28:                                      ; preds = %while.end.split
  %add.lcssa = phi i32 [ %add, %while.end.split ]
  %mul17.lcssa = phi i32 [ %mul17, %while.end.split ]
  %cmp = icmp eq i32 %mul17.lcssa, 1
  %conv = zext i1 %cmp to i32
  %5 = add i32 %mul17.lcssa, 1
  %6 = icmp ult i32 %5, 3
  %div = select i1 %6, i32 %mul17.lcssa, i32 0
  %cmp3 = icmp sgt i32 %div, %conv
  br i1 %cmp3, label %while.body4, label %while.cond1.while.end29_crit_edge

while.cond1.while.end29_crit_edge:                ; preds = %while.end28
  %.lcssa = phi i32 [ %mul17.lcssa, %while.end28 ]
  %add.lcssa50.lcssa = phi i32 [ %add.lcssa, %while.end28 ]
  store i32 %add.lcssa50.lcssa, i32* %a, align 4
  br label %while.end29

while.end29:                                      ; preds = %while.cond1.while.end29_crit_edge, %while.body
  %7 = phi i32 [ %.lcssa, %while.cond1.while.end29_crit_edge ], [ %0, %while.body ]
  br label %while.body
}

; RUN: opt -passes=indvars -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @fn1(i1 %c0, i1 %c1) {
; CHECK-LABEL: @fn1(
entry:
  br i1 %c0, label %for.end.34, label %for.cond.1thread-pre-split

for.cond.loopexit:                                ; preds = %for.end.29, %for.end.7
  %f.lcssa = phi i32 [ %f.1, %for.end.29 ], [ %f.1, %for.end.7 ]
  br i1 %c1, label %for.end.34, label %for.cond.1thread-pre-split

for.cond.1thread-pre-split:                       ; preds = %for.cond.loopexit, %entry
  %f.047 = phi i32 [ %f.lcssa, %for.cond.loopexit ], [ 0, %entry ]
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.cond.1, %for.cond.1thread-pre-split
  br i1 %c1, label %for.cond.4, label %for.cond.1

for.cond.4:                                       ; preds = %for.end.29, %for.cond.1
  %f.1 = phi i32 [ 0, %for.end.29 ], [ %f.047, %for.cond.1 ]
  br label %for.cond.5

for.cond.5:                                       ; preds = %for.cond.5, %for.cond.4
  %h.0 = phi i32 [ 0, %for.cond.4 ], [ %inc, %for.cond.5 ]
  %cmp = icmp slt i32 %h.0, 1
  %inc = add nsw i32 %h.0, 1
  br i1 %cmp, label %for.cond.5, label %for.end.7

for.end.7:                                        ; preds = %for.cond.5
  %g.lcssa = phi i32 [ %h.0, %for.cond.5 ]
  %tobool10 = icmp eq i32 %g.lcssa, 0
  br i1 %tobool10, label %for.end.8, label %for.cond.loopexit

for.end.8:                       ; preds = %for.end.7
  br i1 %c1, label %for.cond.19, label %for.end.29

for.cond.19:                                      ; preds = %for.cond.19, %for.end.8
  br label %for.cond.19

for.end.29:                                       ; preds = %for.end.8
  %tobool30 = icmp eq i32 %f.1, 0
  br i1 %tobool30, label %for.cond.4, label %for.cond.loopexit

for.end.34:                                       ; preds = %for.cond.loopexit, %entry
  ret void
}

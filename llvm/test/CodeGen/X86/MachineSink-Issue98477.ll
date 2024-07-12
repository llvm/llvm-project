; RUN: llc < %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(i1 %tobool.not, i32 %0) {
entry:
  br i1 %tobool.not, label %if.end13, label %j.preheader

  j.preheader:       ; preds = %if.end13, %entry
  %h.0.ph = phi i32 [ 1, %entry ], [ 0, %if.end13 ]
  br label %j

  j:                 ; preds = %if.then4, %j.preheader
  %1 = phi i32 [ %div2, %if.then4 ], [ 0, %j.preheader ]
  %rem1 = srem i32 1, %0
  %cmp = icmp slt i32 %1, 0
  %or.cond = select i1 false, i1 true, i1 %cmp
  br i1 %or.cond, label %if.then4, label %if.end9

  if.then4:          ; preds = %j
  %div2 = sdiv i32 1, 0
  %rem5 = srem i32 1, %h.0.ph
  br i1 %tobool.not, label %if.end9, label %j

  if.end9:           ; preds = %if.then4, %j
  %2 = phi i32 [ 0, %j ], [ %rem5, %if.then4 ]
  %tobool10.not = icmp eq i32 %2, 0
  br i1 %tobool10.not, label %if.end13, label %while.body.lr.ph

  while.body.lr.ph:  ; preds = %if.end9
  ret i32 %rem1

  if.end13:          ; preds = %if.end9, %entry
  br label %j.preheader
}

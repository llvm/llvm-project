; RUN: opt -passes=indvars,loop-idiom,loop-deletion -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Checking for a crash

define void @f(ptr %a) {
; CHECK-LABEL: @f(
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %for.cond, %entry
  %iv = phi i32 [ 0, %entry ], [ %add, %for.inc ], [ %iv, %for.cond ]
  %add = add nsw i32 %iv, 1
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
  br i1 undef, label %for.cond, label %for.inc

for.inc:                                          ; preds = %for.cond
  br i1 undef, label %for.cond, label %for.end

for.end:                                          ; preds = %for.inc
  ret void
}

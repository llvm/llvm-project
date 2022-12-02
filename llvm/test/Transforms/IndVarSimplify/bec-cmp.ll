; RUN: opt -S -passes=indvars < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @foo(i32* nocapture %a, i32* nocapture readonly %b, i32 signext %n) #0 {
entry:

; CHECK-LABEL: @foo

  %cmp.10 = icmp sgt i32 %n, 0
  br i1 %cmp.10, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:              ; preds = %for.inc
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.for.cond.cleanup_crit_edge, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.011 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %cmp1 = icmp sgt i32 %i.011, %n
  br i1 %cmp1, label %if.then, label %for.inc

; CHECK-NOT: br i1 %cmp1, label %if.then, label %for.inc
; CHECK: br i1 false, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %idxprom = sext i32 %i.011 to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, 1
  %arrayidx3 = getelementptr inbounds i32, i32* %a, i64 %idxprom
  store i32 %add, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nsw i32 %i.011, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %for.body, label %for.cond.for.cond.cleanup_crit_edge
}

attributes #0 = { nounwind }


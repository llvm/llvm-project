; RUN: opt < %s -S -loop-flatten -verify-loop-info -verify-dom-info -verify-scev -verify | FileCheck %s

; CHECK-LABEL: @main

define dso_local void @main() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.end, %entry
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.cond
  %a.03 = phi i32 [ 0, %for.cond ], [ %inc, %for.inc ]
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %0 = add i32 %a.03, 1
  %cmp = icmp slt i32 %0, 10
  %inc = add nsw i32 %a.03, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  br label %for.cond
}

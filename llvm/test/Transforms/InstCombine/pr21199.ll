; do not replace a 'select' with 'or' in 'select - cmp - br' sequence
; RUN: opt -instcombine -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @f(i32)

define void @test(i32 %len) {
entry:
  %cmp = icmp ult i32 %len, 8
  %cond = select i1 %cmp, i32 %len, i32 8
  %cmp11 = icmp ult i32 0, %cond
  br i1 %cmp11, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  tail call void @f(i32 %cond)
  %inc = add i32 %i.02, 1
  %cmp1 = icmp ult i32 %inc, %cond
  br i1 %cmp1, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: select
}

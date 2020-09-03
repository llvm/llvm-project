; RUN: opt -S -indvars < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @use(i1)

define void @f(i32 %x) {
; CHECK-LABEL: @f(
 entry:
  %conv = sext i32 %x to i64
  %sub = add i64 %conv, -1
  %ec = icmp sgt i32 %x, 0
  br i1 %ec, label %loop, label %leave

 loop:
; CHECK: loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.inc, %loop ]
  %iv.inc = add i64 %iv, 1
  %cmp = icmp slt i64 %iv, %sub
  call void @use(i1 %cmp)
; CHECK: call void @use(i1 %cmp)
; CHECK-NOT: call void @use(i1 true)

  %be.cond = icmp slt i64 %iv.inc, %conv
  br i1 %be.cond, label %loop, label %leave

 leave:
  ret void
}

; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f_0() {
; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'f_0':
; CHECK: Loop %for.body: backedge-taken count is 5
; CHECK: Loop %for.body: constant max backedge-taken count is 5
; CHECK: Loop %for.body: Predicated backedge-taken count is 5

entry:
  br label %for.body

for.body:
  %i.05 = phi i32 [ 32, %entry ], [ %div4, %for.body ]
  tail call void @dummy()
  %div4 = lshr i32 %i.05, 1
  %cmp = icmp eq i32 %div4, 0
  br i1 %cmp, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

declare void @dummy()

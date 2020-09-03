; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @foo
; CHECK-LABEL: entry
entry:
  %tobool = icmp ne i32 %cond1, 0
  br i1 %tobool, label %bb.f1, label %bb.f2

bb.f1:
  call void @f1()
  br label %bb.cond2
; Verify that we branch on cond2 without checking tobool again.
; CHECK:      call void @f1()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %exit, label %bb.f3

bb.f2:
  call void @f2()
  br label %bb.cond2
; Verify that we branch on cond2 without checking tobool again.
; CHECK:      call void @f2()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %exit, label %bb.f4

bb.cond2:
  %tobool1 = icmp eq i32 %cond2, 0
  br i1 %tobool1, label %exit, label %bb.cond1again

; Verify that we eliminate this basic block.
; CHECK-NOT: bb.cond1again:
bb.cond1again:
  br i1 %tobool, label %bb.f3, label %bb.f4

bb.f3:
  call void @f3()
  br label %exit

bb.f4:
  call void @f4()
  br label %exit

exit:
  ret void
}

declare void @f1() local_unnamed_addr

declare void @f2() local_unnamed_addr

declare void @f3() local_unnamed_addr

declare void @f4() local_unnamed_addr

; RUN: opt < %s -jump-threading -S -verify | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 0, align 4

define void @foo(i32 %cond1, i32 %cond2) {
; CHECK-LABEL: @foo
; CHECK-LABEL: entry
entry:
  %tobool = icmp eq i32 %cond1, 0
  br i1 %tobool, label %bb.cond2, label %bb.f1

bb.f1:
  call void @f1()
  br label %bb.cond2
; Verify that we branch on cond2 without checking ptr.
; CHECK:      call void @f1()
; CHECK-NEXT: icmp eq i32 %cond2, 0
; CHECK-NEXT: label %bb.f4, label %bb.f2

bb.cond2:
  %ptr = phi i32* [ null, %bb.f1 ], [ @a, %entry ]
  %tobool1 = icmp eq i32 %cond2, 0
  br i1 %tobool1, label %bb.file, label %bb.f2
; Verify that we branch on cond2 without checking ptr.
; CHECK:      icmp eq i32 %cond2, 0
; CHECK-NEXT: label %bb.f3, label %bb.f2

bb.f2:
  call void @f2()
  br label %exit

; Verify that we eliminate this basic block.
; CHECK-NOT: bb.file:
bb.file:
  %cmp = icmp eq i32* %ptr, null
  br i1 %cmp, label %bb.f4, label %bb.f3

bb.f3:
  call void @f3()
  br label %exit

bb.f4:
  call void @f4()
  br label %exit

exit:
  ret void
}

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()

; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s
; LSR used to pick a sub-optimal solution due to the target responding
; conservatively to isLegalAddImmediate for negative values.

declare void @foo(i32)

define void @test(i32 %px) {
; CHECK_LABEL: test:
; CHECK_LABEL: %entry
; CHECK: subs
; CHECK-NEXT: csel
entry:
  %sub = add nsw i32 %px, -1
  %cmp = icmp slt i32 %px, 1
  %.sub = select i1 %cmp, i32 0, i32 %sub
  br label %for.body

for.body:
; CHECK_LABEL: %for.body
; CHECK:  cmp
; CHECK-NEXT:  b.eq
; CHECK-LABEL:  %if.then3
  %x.015 = phi i32 [ %inc, %for.inc ], [ %.sub, %entry ]
  %cmp2 = icmp eq i32 %x.015, %px
  br i1 %cmp2, label %for.inc, label %if.then3

if.then3:
  tail call void @foo(i32 %x.015)
  br label %for.inc

for.inc:
; CHECK_LABEL: %for.inc
; CHECK:  add
; CHECK-NEXT:  cmp
; CHECK:  b.le
; CHECK_LABEL: %for.cond.cleanup
  %inc = add nsw i32 %x.015, 1
  %cmp1 = icmp sgt i32 %x.015, %px
  br i1 %cmp1, label %for.cond.cleanup.loopexit, label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

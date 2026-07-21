; RUN: opt -passes=dead-branch-elim -S %s | FileCheck %s
; The pass must be idempotent:
; RUN: opt -passes=dead-branch-elim,dead-branch-elim -S %s | FileCheck %s

; @multi_block: the dead body spans several blocks (a nested if); cutting the
; branch edge must remove the whole region, including the nested limit
; modifications.
; CHECK-LABEL: define i32 @multi_block()
; CHECK: loop:
; CHECK-NOT: if.then:
; CHECK-NOT: deep:
; CHECK-NOT: if.end:
; CHECK-NOT: icmp eq
; CHECK: latch:
; CHECK: exit:
define i32 @multi_block() {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %b = phi i32 [ 0, %entry ], [ %b.next, %latch ]
  %limit = phi i32 [ 100, %entry ], [ %limit.next, %latch ]
  %cmp.inner = icmp eq i32 %b, %limit
  br i1 %cmp.inner, label %if.then, label %latch

if.then:
  %deep.cmp = icmp sgt i32 %a, 5
  br i1 %deep.cmp, label %deep, label %if.end

deep:
  %limit.inc2 = add nsw i32 %limit, 2
  br label %if.end

if.end:
  %limit.new = phi i32 [ %limit.inc2, %deep ], [ %limit, %if.then ]
  %limit.inc1 = add nsw i32 %limit.new, 1
  br label %latch

latch:
  %limit.next = phi i32 [ %limit.inc1, %if.end ], [ %limit, %loop ]
  %a.next = add nsw i32 %a, 1
  %b.next = add nsw i32 %b, 1
  %cmp.outer = icmp slt i32 %a.next, %limit.next
  br i1 %cmp.outer, label %loop, label %exit

exit:
  ret i32 %b.next
}

; @cascade: 'tail' bumps limit and is reachable from two dead sources.
; Neither is provable in isolation; only the fixed point that assumes both
; dead at once can remove them.
; CHECK-LABEL: define i32 @cascade()
; CHECK: loop:
; CHECK-NOT: body1:
; CHECK-NOT: tail:
; CHECK-NOT: icmp sgt
; CHECK-NOT: icmp eq
; CHECK: mid:
; CHECK: latch:
; CHECK: exit:
define i32 @cascade() {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %b = phi i32 [ 0, %entry ], [ %b.next, %latch ]
  %limit = phi i32 [ 100, %entry ], [ %limit.next, %latch ]
  %cmp1 = icmp sgt i32 %a, 1000
  br i1 %cmp1, label %body1, label %mid

body1:
  br label %tail

mid:
  %cmp2 = icmp eq i32 %b, %limit
  br i1 %cmp2, label %tail, label %latch

tail:
  %limit.bump = add nsw i32 %limit, 1
  br label %latch

latch:
  %limit.next = phi i32 [ %limit.bump, %tail ], [ %limit, %mid ]
  %a.next = add nsw i32 %a, 1
  %b.next = add nsw i32 %b, 1
  %cmp.out = icmp slt i32 %a.next, 100
  br i1 %cmp.out, label %loop, label %exit

exit:
  ret i32 %limit.next
}

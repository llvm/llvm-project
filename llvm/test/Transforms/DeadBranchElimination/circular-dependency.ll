; RUN: opt -passes=dead-branch-elim -S %s | FileCheck %s

; The motivating example (rotated loop form):
;   int a = 0, b = 0, limit = 100;
;   while (a < limit) {
;     if (b == limit)   // unreachable: a == b always, and a < limit here
;       limit += 1;     // ...but it modifies limit (circular dependency)
;     a++; b++;
;   }
; The 'loop' -> 'if.then' edge must be removed. @keep (whose branch is
; genuinely taken) must stay intact.

; CHECK-LABEL: define i32 @run()
; CHECK: loop:
; CHECK-NOT: if.then:
; CHECK-NOT: icmp eq
; CHECK: br label %latch
; CHECK: latch:
; CHECK: br i1 %cmp.outer, label %loop, label %exit
; CHECK: exit:
define i32 @run() {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %b = phi i32 [ 0, %entry ], [ %b.next, %latch ]
  %limit = phi i32 [ 100, %entry ], [ %limit.next, %latch ]
  %cmp.inner = icmp eq i32 %b, %limit
  br i1 %cmp.inner, label %if.then, label %latch

if.then:
  %limit.inc = add nsw i32 %limit, 1
  br label %latch

latch:
  %limit.next = phi i32 [ %limit.inc, %if.then ], [ %limit, %loop ]
  %a.next = add nsw i32 %a, 1
  %b.next = add nsw i32 %b, 1
  %cmp.outer = icmp slt i32 %a.next, %limit.next
  br i1 %cmp.outer, label %loop, label %exit

exit:
  ret i32 %b.next
}

; CHECK-LABEL: define i32 @keep()
; CHECK: loop:
; CHECK: br i1 %cmp.inner, label %if.then, label %latch
; CHECK: if.then:
; CHECK: latch:
define i32 @keep() {
entry:
  br label %loop

loop:
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %latch ]
  %cmp.inner = icmp eq i32 %a, 50
  br i1 %cmp.inner, label %if.then, label %latch

if.then:
  %sum.inc = add nsw i32 %sum, %a
  br label %latch

latch:
  %sum.next = phi i32 [ %sum.inc, %if.then ], [ %sum, %loop ]
  %a.next = add nsw i32 %a, 1
  %cmp.outer = icmp slt i32 %a.next, 100
  br i1 %cmp.outer, label %loop, label %exit

exit:
  ret i32 %sum.next
}

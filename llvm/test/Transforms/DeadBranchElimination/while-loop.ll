; RUN: opt -passes=dead-branch-elim -S %s | FileCheck %s

; The motivating example exactly as clang -O0 + mem2reg emits it: an
; UNROTATED while loop, guard in the header, inner check in the body.
; Proving b != limit needs the dominating guard a < limit (b's global range
; includes the value reached on the exit iteration), i.e. a context-sensitive
; SCEV query.

; CHECK-LABEL: define i32 @run()
; CHECK: header:
; CHECK: body:
; CHECK-NOT: if.then:
; CHECK-NOT: icmp eq
; CHECK: br label %latch
; CHECK: latch:
; CHECK: exit:
define i32 @run() {
entry:
  br label %header

header:
  %limit = phi i32 [ 100, %entry ], [ %limit.next, %latch ]
  %b = phi i32 [ 0, %entry ], [ %b.next, %latch ]
  %a = phi i32 [ 0, %entry ], [ %a.next, %latch ]
  %guard = icmp slt i32 %a, %limit
  br i1 %guard, label %body, label %exit

body:
  %cmp.inner = icmp eq i32 %b, %limit
  br i1 %cmp.inner, label %if.then, label %latch

if.then:
  %limit.inc = add nsw i32 %limit, 1
  br label %latch

latch:
  %limit.next = phi i32 [ %limit.inc, %if.then ], [ %limit, %body ]
  %a.next = add nsw i32 %a, 1
  %b.next = add nsw i32 %b, 1
  br label %header

exit:
  ret i32 %b
}

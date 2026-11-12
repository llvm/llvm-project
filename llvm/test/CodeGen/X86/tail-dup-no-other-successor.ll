; RUN: llc -O3 -o - %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @effect(i32);

; After the loop gets laid out, loop.end is the only successor, but can't be
; laid out because of the CFG dependency from top.fakephi. The calculations show
; that it isn't profitable to tail-duplicate in this case, because of the
; effects on fallthrough from %loop.end
; CHECK-LABEL: {{^}}no_successor_still_no_taildup:
; CHECK: %entry
; CHECK: %loop.top
; CHECK: %loop.latch
; CHECK: %loop.end
; CHECK: %false
; CHECK: %ret
; CHECK: %top.fakephi
define void @no_successor_still_no_taildup (i32 %count, i32 %key) {
entry:
  br label %loop.top

loop.top:
  %i.loop.top = phi i32 [ %count, %entry ], [ %i.latch, %loop.latch ]
  %cmp.top = icmp eq i32 %i.loop.top, %key
  call void @effect(i32 0)
  br i1 %cmp.top, label %top.fakephi, label %loop.latch, !prof !1

loop.latch:
  %i.latch = sub i32 %i.loop.top, 1
  %cmp.latch = icmp eq i32 %i.latch, 0
  call void @effect(i32 1)
  br i1 %cmp.top, label %loop.top, label %loop.end, !prof !2

top.fakephi:
  call void @effect(i32 2)
  br label %loop.end

loop.end:
  %cmp.end = icmp eq i32 %count, 0
  br i1 %cmp.end, label %ret, label %false, !prof !3

false:
  call void @effect(i32 4)
  br label %ret

ret:
  ret void
}

!1 = !{!"branch_weights", i32 1, i32 1}
!2 = !{!"branch_weights", i32 5, i32 1}
!3 = !{!"branch_weights", i32 1, i32 2}

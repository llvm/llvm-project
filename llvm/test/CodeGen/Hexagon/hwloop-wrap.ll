; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s

; We shouldn't generate a hardware loop in this case because the initial
; value may be zero, which means the endloop instruction will not decrement
; the loop counter, and the loop will execute only once.

; CHECK-NOT: loop0

define void @foo(i32 %count, i32 %v) #0 {
entry:
  br label %do.body

do.body:
  %count.addr.0 = phi i32 [ %count, %entry ], [ %dec, %do.body ]
  tail call void asm sideeffect "nop", ""() #1
  %dec = add i32 %count.addr.0, -1
  %cmp = icmp eq i32 %dec, 0
  br i1 %cmp, label %do.end, label %do.body

do.end:
  ret void
}

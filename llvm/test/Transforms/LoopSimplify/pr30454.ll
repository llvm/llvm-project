; RUN: opt < %s -passes='lcssa,loop-mssa(licm)' -S | FileCheck %s
; PR30454
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i8 @bar()

; Test that we preserve LCSSA form when removing edges from unreachable blocks.
; CHECK-LABEL: @foo
define void @foo() {
entry:
  br label %for.cond

for.cond:
  %x = phi i8 [ undef, %entry ], [ %y, %for.latch ]
  br i1 undef, label %for.latch, label %exit

; CHECK:      unreachable.bb:
; CHECK-NEXT:   unreachable
unreachable.bb:
  br i1 undef, label %exit, label %for.latch

for.latch:
  %y = call i8 @bar()
  br label %for.cond

; CHECK:      exit:
; CHECK-NEXT:   %x.lcssa = phi i8 [ %x, %for.cond ]
exit:
  %z = zext i8 %x to i32
  ret void
}

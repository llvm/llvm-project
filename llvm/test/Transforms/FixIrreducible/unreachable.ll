; RUN: opt %s -fix-irreducible -S -o - | FileCheck %s

; CHECK-LABEL: @unreachable(
; CHECK: entry:
; CHECK-NOT: irr.guard:
define void @unreachable(i32 %n) {
entry:
  br label %loop.body

loop.body:
  br label %inner.block

unreachable.block:
  br label %inner.block

inner.block:
  br i1 undef, label %loop.exit, label %loop.latch

loop.latch:
  br label %loop.body

loop.exit:
  ret void
}

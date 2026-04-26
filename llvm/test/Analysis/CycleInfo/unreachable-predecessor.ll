; RUN: opt < %s -disable-output -passes='verify<cycles>,print<cycles>' 2>&1 | FileCheck %s
; CHECK-LABEL: CycleInfo for function: unreachable
; CHECK:    depth=1: entries(loop.body) loop.latch inner.block
define void @unreachable(i32 %n, i1 %arg) {
entry:
  br label %loop.body

loop.body:
  br label %inner.block

; This branch should not cause %inner.block to appear as an entry.
unreachable.block:
  br label %inner.block

inner.block:
  br i1 %arg, label %loop.exit, label %loop.latch

loop.latch:
  br label %loop.body

loop.exit:
  ret void
}

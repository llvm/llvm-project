; NOTE: Do not autogenerate
; RUN: opt < %s -fix-irreducible --verify-loop-info -S | FileCheck %s
; RUN: opt < %s -passes='fix-irreducible,verify<loops>' -S | FileCheck %s
; RUN: opt < %s -passes='verify<loops>,fix-irreducible,verify<loops>' -S | FileCheck %s

; CHECK-LABEL: @unreachable(
; CHECK: entry:
; CHECK-NOT: irr.guard:
define void @unreachable(i32 %n, i1 %arg) {
entry:
  br label %loop.body

loop.body:
  br label %inner.block

unreachable.block:
  br label %inner.block

inner.block:
  br i1 %arg, label %loop.exit, label %loop.latch

loop.latch:
  br label %loop.body

loop.exit:
  ret void
}

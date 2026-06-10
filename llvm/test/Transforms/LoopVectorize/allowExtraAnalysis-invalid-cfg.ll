; RUN: opt -S -passes=loop-vectorize -pass-remarks-output=%t.yaml < %s | FileCheck %s

; Enabling remarks forces extra legality analysis on loops with unexpected CFG.
; Ensure LV does not crash due to the loop having multiple latches.
; A normal branch gets canonicalized to a unique latch before LV runs, so use
; indirectbr to keep the invalid CFG shape.

; CHECK-LABEL: define void @multiple_latches_indirectbr(
define void @multiple_latches_indirectbr() {
entry:
  indirectbr ptr null, [label %loop]

loop:
  br i1 false, label %bb, label %loop

bb:
  br label %loop
}

; CHECK-LABEL: define void @multiple_latches_and_predecessors_indirectbr(
define void @multiple_latches_and_predecessors_indirectbr() {
entry:
  indirectbr ptr null, [label %loop, label %side]

side:
  br label %loop

loop:
  %x = load i64, ptr null, align 8
  br i1 false, label %back, label %loop

back:
  br label %loop
}

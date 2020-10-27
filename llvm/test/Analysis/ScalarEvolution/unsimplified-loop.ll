; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; This loop has no preheader, multiple backedges, etc., but ScalarEvolution
; should still be able to analyze it.

; CHECK: %i = phi i64 [ 5, %entry ], [ 5, %alt ], [ %i.next, %loop.a ], [ %i.next, %loop.b ]
; CHECK-NEXT: -->  {5,+,1}<%loop>

define void @foo(i1 %p, i1 %q, i1 %s, i1 %u) {
entry:
  br i1 %p, label %loop, label %alt

alt:
  br i1 %s, label %loop, label %exit

loop:
  %i = phi i64 [ 5, %entry ], [ 5, %alt ], [ %i.next, %loop.a ], [ %i.next, %loop.b ]
  %i.next = add i64 %i, 1
  br i1 %q, label %loop.a, label %loop.b

loop.a:
  br label %loop

loop.b:
  br i1 %u, label %loop, label %exit

exit:
  ret void
}

; RUN: opt -passes=loop-vectorize -force-vector-width=4 \
; RUN:   -force-vector-interleave=1 -S < %s | FileCheck %s

target triple = "x86_64-apple-macosx10.12.0"

define { i64, i64 } @conditional_induction_liveout(i64 %len, i64 %x) {
entry:
  br label %loop

loop:
  %at = phi i64 [ 1, %entry ], [ %at.next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  %acc.next = xor i64 %acc, %x
  %more = icmp ult i64 %at, %len
  %inc = zext i1 %more to i64
  %at.next = add i64 %at, %inc
  br i1 %more, label %loop, label %exit

exit:
  %a = insertvalue { i64, i64 } poison, i64 %acc.next, 0
  %b = insertvalue { i64, i64 } %a, i64 %at.next, 1
  ret { i64, i64 } %b
}

; The index only advances while the loop continues. Its exiting value is not
; the affine induction value after the final vector lane, so this loop must
; remain scalar until the vectorizer can preserve that conditional update.
; CHECK-LABEL: @conditional_induction_liveout(
; CHECK-NOT: vector.body
; CHECK: %at.next = add i64 %at, %inc
; CHECK-NOT: vector.body

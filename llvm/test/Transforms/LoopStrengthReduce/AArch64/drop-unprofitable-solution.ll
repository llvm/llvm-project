; RUN: llc -mtriple=aarch64 -stop-after=loop-reduce < %s | FileCheck %s

; Keep the cheaper baseline induction variable instead of replacing it with an
; LSR solution that requires an extra base adjustment.

; CHECK-LABEL: define i64 @scan(
; CHECK: %index = phi i64 [ %index.next, %loop ], [ 0, %entry ]
; CHECK: %index.next = add i64 %index, 1
; CHECK-NOT: %lsr.iv

define i64 @scan(ptr %base, i64 %offset, i1 %condition) {
entry:
  br label %loop

loop:
  %index = phi i64 [ %index.next, %loop ], [ 0, %entry ]
  %index.next = add i64 %index, 1
  %ptr = getelementptr i8, ptr %base, i64 %offset
  br i1 %condition, label %loop, label %exit

exit:
  ret i64 %index
}

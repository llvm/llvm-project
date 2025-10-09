; REQUIRES: asserts
; RUN: opt -p loop-vectorize -debug-only=loop-vectorize --disable-output -S %s 2>&1 | FileCheck %s


; CHECK-LABEL: LV: Checking a loop in 'latch_exit_cannot_compute_btc_due_to_step'
; CHECK: 	   LV: Did not find one integer induction var.
; CHECK-NEXT:  LV: Not vectorizing: Cannot vectorize uncountable loop.
; CHECK-NEXT:  LV: Interleaving disabled by the pass manager
; CHECK-NEXT:  LV: Not vectorizing: Cannot prove legality.

; CHECK-LABEL: LV: Checking a loop in 'header_exit_cannot_compute_btc_due_to_step'
; CHECK:       LV: Found an induction variable.
; CHECK-NEXT:  LV: Did not find one integer induction var.
; CHECK-NEXT:  LV: Not vectorizing: Cannot vectorize uncountable loop.
; CHECK-NEXT:  LV: Interleaving disabled by the pass manager
; CHECK-NEXT:  LV: Not vectorizing: Cannot prove legality.

; CHECK-NOT: vector.body
define void @latch_exit_cannot_compute_btc_due_to_step(ptr %dst, i64 %step) {
entry:
  br label %loop

loop:                                   ; preds = %loop, %for.cond.us
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i64 %iv, %step
  %gep = getelementptr i8, ptr %dst, i64 %iv
  store i8 0, ptr %gep, align 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %loop, label %exit

exit:
  ret void
}

define void @header_exit_cannot_compute_btc_due_to_step(ptr %dst, i64 %step) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.next = add i64 %iv, %step
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %loop.latch, label %exit

loop.latch:
  %gep = getelementptr i8, ptr %dst, i64 %iv
  store i8 0, ptr %gep, align 1
  br label %loop.header

exit:
  ret void
}

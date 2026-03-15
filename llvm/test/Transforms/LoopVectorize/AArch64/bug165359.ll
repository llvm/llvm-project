; RUN: opt < %s -passes=loop-vectorize -S -pass-remarks-analysis=loop-vectorize -disable-output &> %t
; RUN: cat %t | FileCheck --check-prefix=CHECK-REMARKS %s

; CHECK-REMARKS: remark: <unknown>:0:0: Recipe with invalid costs prevented vectorization at VF=(vscale x 1): fadd

target triple = "aarch64-unknown-linux-gnu"

define double @reduce_fail(i64 %loop_count, double %d0, ptr %ptr1) #0 {
entry:
  %d1 = load double, ptr %ptr1
  br label %loop

loop:
  %acc0 = phi double [ %fadd0, %loop ], [ %d0, %entry ]
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %fadd0 = fadd double %acc0, %d1
  %iv.next = add nsw nuw i64 %iv, 1
  %exit_cond = icmp eq i64 %iv.next, %loop_count
  br i1 %exit_cond, label %loopexit, label %loop

loopexit:
  ret double %fadd0
}

attributes #0 = { "target-features"="+sve" }

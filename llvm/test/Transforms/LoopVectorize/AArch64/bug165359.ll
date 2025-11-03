; RUN: opt < %s -passes=loop-vectorize -S -pass-remarks=loop-vectorize -debug-only=loop-vectorize &> %t
; RUN: cat %t | FileCheck --check-prefix=CHECK-REMARKS %s

; CHECK-REMARKS: LV: Recipe with invalid costs prevented vectorization at VF=(vscale x 1): fadd.

target triple = "aarch64-unknown-linux-gnu"

define void @reduce_fail(i64 %loop_count, ptr %ptr0, ptr noalias %ptr1) #0 {
entry:
  %d1 = load double, ptr %ptr1
  %d0 = load double, ptr %ptr0
  br label %loop

loop:
  %acc0 = phi double [ %fadd0, %loop ], [ %d0, %entry ]
  %counter = phi i64 [ %counter_updated, %loop ], [ %loop_count, %entry ]
  %fadd0 = fadd double %acc0, %d1
  %counter_updated = add nsw i64 %counter, -1
  %exit_cond = icmp samesign ugt i64 %counter, 1
  br i1 %exit_cond, label %loop, label %loopexit

loopexit:
  store double %fadd0, ptr %ptr1
  ret void
}

attributes #0 = { "target-features"="+sve" }
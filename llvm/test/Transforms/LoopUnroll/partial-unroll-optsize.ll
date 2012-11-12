; RUN: opt < %s -S -loop-unroll -unroll-allow-partial | FileCheck %s
; Loop size = 3, when the function has the optsize attribute, the
; OptSizeUnrollThreshold, i.e. 50, is used, hence the loop should be unrolled
; by 16 times because 3 * 16 < 50.
define void @unroll_opt_for_size() nounwind optsize {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; CHECK:      add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: icmp

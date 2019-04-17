; RUN: opt < %s -S -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -loop-unroll | FileCheck %s
define void @unroll_default() nounwind {
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

; CHECK-LABEL: @unroll_default
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
; CHECK-NEXT: add
; CHECK-NEXT: icmp


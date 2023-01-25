; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=cortex-a57 | FileCheck %s

; Partial unroll 8 times for this loop.
define void @unroll1() nounwind {
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
; CHECK-NEXT: icmp

; Partial unroll 16 times for this loop.
define void @unroll2() nounwind {
entry:
  br label %loop1

loop1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch ]
  br label %loop2.header

loop2.header:
  br label %loop2

loop2:
  %iv2 = phi i32 [ 0, %loop2.header ], [ %inc2, %loop2 ]
  %inc2 = add i32 %iv2, 1
  %exitcnd2 = icmp uge i32 %inc2, 1024
  br i1 %exitcnd2, label %exit2, label %loop2

exit2:
  br label %loop1.latch

loop1.latch:
  %inc1 = add i32 %iv1, 1
  %exitcnd1 = icmp uge i32 %inc1, 1024
  br i1 %exitcnd2, label %exit, label %loop1

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

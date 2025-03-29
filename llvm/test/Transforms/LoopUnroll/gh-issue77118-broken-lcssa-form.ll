; RUN: opt -passes=loop-unroll -unroll-peel-count=2 -S -disable-output -debug-only=loop-unroll < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

define void @test() {
; CHECK-LABEL: Loop Unroll: F[test] Loop %loop3
; CHECK-NEXT:    Loop Size = 7
; CHECK-NEXT:  PEELING loop %loop3 with iteration count 2!
; CHECK-NEXT:  Loop Unroll: F[test] Loop %loop2
; CHECK-NEXT:    Loop Size = 28
; CHECK-NEXT:  PEELING loop %loop2 with iteration count 2!
; CHECK-NEXT:  Loop Unroll: F[test] Loop %loop4
; CHECK-NEXT:    Loop Size = 3
; CHECK-NEXT:  PEELING loop %loop4 with iteration count 2!
; CHECK-NEXT:  Loop Unroll: F[test] Loop %loop1
; CHECK-NEXT:    Loop Size = 95
; CHECK-NEXT:  PEELING loop %loop1 with iteration count 2!
entry:
  br label %loop1

loop1:
  %phi = phi i32 [ 1, %entry ], [ 0, %loop1.latch ]
  br label %loop2

loop2:
  %phi3 = phi i64 [ 0, %loop1 ], [ %sext, %loop2.latch ]
  br label %loop3

loop3:
  %phi5 = phi i64 [ %phi3, %loop2 ], [ %sext, %loop3.latch ]
  %phi6 = phi i32 [ 1, %loop2 ], [ %add10, %loop3.latch ]
  %trunc = trunc i64 %phi5 to i32
  br i1 true, label %loop3.latch, label %exit

loop3.latch:
  %add = add i32 1, %phi
  %sext = sext i32 %add to i64
  %add10 = add i32 %phi6, 1
  %icmp = icmp ugt i32 %add10, 2
  br i1 %icmp, label %loop2.latch, label %loop3

loop2.latch:
  br i1 false, label %loop4.preheader, label %loop2

loop4.preheader:
  br label %loop4

loop4:
  br i1 false, label %loop1.latch, label %loop4

loop1.latch:
  br label %loop1

exit:
  %phi8 = phi i32 [ %trunc, %loop3 ]
  ret void
}

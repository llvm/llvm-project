; RUN: opt < %s -S -passes="default<O2>" -unroll-runtime=true -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=O2
; RUN: opt < %s -S -passes="default<O3>" -unroll-runtime=true -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=O3
; RUN: opt < %s -S -passes="default<Os>" -unroll-runtime=true -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=Os
; RUN: opt < %s -S -passes="default<Oz>" -unroll-runtime=true -unroll-threshold-default=0 -unroll-threshold-aggressive=300 | FileCheck %s -check-prefix=Oz

; Check that Os and Oz are optimized like O2, not like O3. To easily highlight
; the behavior, we artificially disable unrolling for anything but O3 by setting
; the default threshold to 0.

; O3:     loop2.preheader
; O2-NOT: loop2.preheader
; Os-NOT: loop2.preheader
; Oz-NOT: loop2.preheader

define void @unroll(i32 %iter, i32* %addr1, i32* %addr2) nounwind {
entry:
  br label %loop1

loop1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch ]
  %offset1 = getelementptr i32, i32* %addr1, i32 %iv1
  store i32 %iv1, i32* %offset1, align 4
  br label %loop2.header

loop2.header:
  %e = icmp uge i32 %iter, 1
  br i1 %e, label %loop2, label %exit2

loop2:
  %iv2 = phi i32 [ 0, %loop2.header ], [ %inc2, %loop2 ]
  %offset2 = getelementptr i32, i32* %addr2, i32 %iv2
  store i32 %iv2, i32* %offset2, align 4
  %inc2 = add i32 %iv2, 1
  %exitcnd2 = icmp uge i32 %inc2, %iter
  br i1 %exitcnd2, label %exit2, label %loop2

exit2:
  br label %loop1.latch

loop1.latch:
  %inc1 = add i32 %iv1, 1
  %exitcnd1 = icmp uge i32 %inc1, 1024
  br i1 %exitcnd1, label %exit, label %loop1

exit:
  ret void
}

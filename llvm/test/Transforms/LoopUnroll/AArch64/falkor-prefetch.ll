; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=falkor | FileCheck %s
; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=falkor -enable-falkor-hwpf-unroll-fix=0 | FileCheck %s --check-prefix=NOHWPF

; Check that loop unroller doesn't exhaust HW prefetcher resources.

; Partial unroll 2 times for this loop on falkor instead of 4.
; NOHWPF-LABEL: @unroll1(
; NOHWPF-LABEL: loop:
; NOHWPF-NEXT: phi
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: icmp
; NOHWPF-NEXT: br
; NOHWPF-NEXT-LABEL: exit:
;
; CHECK-LABEL: @unroll1(
; CHECK-LABEL: loop:
; CHECK-NEXT: phi
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: icmp
; CHECK-NEXT: br
; CHECK-NEXT-LABEL: exit:
define void @unroll1(ptr %p, ptr %p2) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]

  %gep = getelementptr inbounds i32, ptr %p, i32 %iv
  %load = load volatile i32, ptr %gep

  %gep2 = getelementptr inbounds i32, ptr %p2, i32 %iv
  %load2 = load volatile i32, ptr %gep2

  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; Partial unroll 4 times for this loop on falkor instead of 8.
; NOHWPF-LABEL: @unroll2(
; NOHWPF-LABEL: loop2:
; NOHWPF-NEXT: phi
; NOHWPF-NEXT: phi
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: getelementptr
; NOHWPF-NEXT: load
; NOHWPF-NEXT: add
; NOHWPF-NEXT: add
; NOHWPF-NEXT: icmp
; NOHWPF-NEXT: br
; NOHWPF-NEXT-LABEL: exit2:
;
; CHECK-LABEL: @unroll2(
; CHECK-LABEL: loop2:
; CHECK-NEXT: phi
; CHECK-NEXT: phi
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: icmp
; CHECK-NEXT: br
; CHECK-NEXT-LABEL: exit2:

define void @unroll2(ptr %p) {
entry:
  br label %loop1

loop1:
  %iv1 = phi i32 [ 0, %entry ], [ %inc1, %loop1.latch ]
  %outer.sum = phi i32 [ 0, %entry ], [ %sum, %loop1.latch ]
  br label %loop2.header

loop2.header:
  br label %loop2

loop2:
  %iv2 = phi i32 [ 0, %loop2.header ], [ %inc2, %loop2 ]
  %sum = phi i32 [ %outer.sum, %loop2.header ], [ %sum.inc, %loop2 ]
  %gep = getelementptr inbounds i32, ptr %p, i32 %iv2
  %load = load i32, ptr %gep
  %sum.inc = add i32 %sum, %load
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


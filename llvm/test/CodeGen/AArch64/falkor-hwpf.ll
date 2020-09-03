; RUN: opt < %s -S -falkor-hwpf-fix -mtriple aarch64 -mcpu=falkor | FileCheck %s
; RUN: opt < %s -S -falkor-hwpf-fix -mtriple aarch64 -mcpu=cortex-a57 | FileCheck %s --check-prefix=NOHWPF

; Check that strided access metadata is added to loads in inner loops when compiling for Falkor.

; CHECK-LABEL: @hwpf1(
; CHECK: load i32, i32* %gep, align 4, !falkor.strided.access !0
; CHECK: load i32, i32* %gep2, align 4, !falkor.strided.access !0

; NOHWPF-LABEL: @hwpf1(
; NOHWPF: load i32, i32* %gep, align 4{{$}}
; NOHWPF: load i32, i32* %gep2, align 4{{$}}
define void @hwpf1(i32* %p, i32* %p2) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]

  %gep = getelementptr inbounds i32, i32* %p, i32 %iv
  %load = load i32, i32* %gep

  %gep2 = getelementptr inbounds i32, i32* %p2, i32 %iv
  %load2 = load i32, i32* %gep2

  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; Check that outer loop strided load isn't marked.
; CHECK-LABEL: @hwpf2(
; CHECK: load i32, i32* %gep, align 4, !falkor.strided.access !0
; CHECK: load i32, i32* %gep2, align 4{{$}}

; NOHWPF-LABEL: @hwpf2(
; NOHWPF: load i32, i32* %gep, align 4{{$}}
; NOHWPF: load i32, i32* %gep2, align 4{{$}}
define void @hwpf2(i32* %p) {
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
  %gep = getelementptr inbounds i32, i32* %p, i32 %iv2
  %load = load i32, i32* %gep
  %sum.inc = add i32 %sum, %load
  %inc2 = add i32 %iv2, 1
  %exitcnd2 = icmp uge i32 %inc2, 1024
  br i1 %exitcnd2, label %exit2, label %loop2

exit2:
  %gep2 = getelementptr inbounds i32, i32* %p, i32 %iv1
  %load2 = load i32, i32* %gep2
  br label %loop1.latch

loop1.latch:
  %inc1 = add i32 %iv1, 1
  %exitcnd1 = icmp uge i32 %inc1, 1024
  br i1 %exitcnd2, label %exit, label %loop1

exit:
  ret void
}


; Check that non-strided load isn't marked.
; CHECK-LABEL: @hwpf3(
; CHECK: load i32, i32* %gep, align 4, !falkor.strided.access !0
; CHECK: load i32, i32* %gep2, align 4{{$}}

; NOHWPF-LABEL: @hwpf3(
; NOHWPF: load i32, i32* %gep, align 4{{$}}
; NOHWPF: load i32, i32* %gep2, align 4{{$}}
define void @hwpf3(i32* %p, i32* %p2) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]

  %gep = getelementptr inbounds i32, i32* %p, i32 %iv
  %load = load i32, i32* %gep

  %gep2 = getelementptr inbounds i32, i32* %p2, i32 %load
  %load2 = load i32, i32* %gep2

  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

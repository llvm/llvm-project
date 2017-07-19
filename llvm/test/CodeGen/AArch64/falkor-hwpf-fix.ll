; RUN: llc < %s -mtriple aarch64 -mcpu=falkor -disable-post-ra | FileCheck %s

; Check that strided load tag collisions are avoided on Falkor.

; CHECK-LABEL: hwpf1:
; CHECK: ldp {{w[0-9]+}}, {{w[0-9]+}}, [x[[BASE:[0-9]+]], #-16]
; CHECK: mov x[[BASE2:[0-9]+]], x[[BASE]]
; CHECK: ldp {{w[0-9]+}}, {{w[0-9]+}}, [x[[BASE2]], #-8]
; CHECK: ldp {{w[0-9]+}}, {{w[0-9]+}}, [x[[BASE3:[0-9]+]]]
; CHECK: mov x[[BASE4:[0-9]+]], x[[BASE3]]
; CHECK: ldp {{w[0-9]+}}, {{w[0-9]+}}, [x[[BASE4]], #8]

define void @hwpf1(i32* %p, i32* %sp, i32* %sp2, i32* %sp3, i32* %sp4) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]

  %gep = getelementptr inbounds i32, i32* %p, i32 %iv
  %load1 = load i32, i32* %gep

  %gep2 = getelementptr inbounds i32, i32* %gep, i32 1
  %load2 = load i32, i32* %gep2

  %add = add i32 %load1, %load2
  %storegep = getelementptr inbounds i32, i32* %sp, i32 %iv
  store i32 %add, i32* %storegep

  %gep3 = getelementptr inbounds i32, i32* %gep, i32 2
  %load3 = load i32, i32* %gep3

  %gep4 = getelementptr inbounds i32, i32* %gep, i32 3
  %load4 = load i32, i32* %gep4

  %add2 = add i32 %load3, %load4
  %storegep2 = getelementptr inbounds i32, i32* %sp2, i32 %iv
  store i32 %add2, i32* %storegep2

  %gep5 = getelementptr inbounds i32, i32* %gep, i32 4
  %load5 = load i32, i32* %gep5

  %gep6 = getelementptr inbounds i32, i32* %gep, i32 5
  %load6 = load i32, i32* %gep6

  %add3 = add i32 %load5, %load6
  %storegep3 = getelementptr inbounds i32, i32* %sp3, i32 %iv
  store i32 %add3, i32* %storegep3

  %gep7 = getelementptr inbounds i32, i32* %gep, i32 6
  %load7 = load i32, i32* %gep7

  %gep8 = getelementptr inbounds i32, i32* %gep, i32 7
  %load8 = load i32, i32* %gep8

  %add4 = add i32 %load7, %load8
  %storegep4 = getelementptr inbounds i32, i32* %sp4, i32 %iv
  store i32 %add4, i32* %storegep4

  %inc = add i32 %iv, 8
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}


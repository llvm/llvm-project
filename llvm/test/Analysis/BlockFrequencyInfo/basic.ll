; RUN: opt < %s -analyze -block-freq | FileCheck %s

define i32 @test1(i32 %i, i32* %a) {
; CHECK: Printing analysis {{.*}} for function 'test1'
; CHECK: entry = 1024
entry:
  br label %body

; Loop backedges are weighted and thus their bodies have a greater frequency.
; CHECK: body = 31744
body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %0 = load i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body

; CHECK: exit = 1024
exit:
  ret i32 %sum
}

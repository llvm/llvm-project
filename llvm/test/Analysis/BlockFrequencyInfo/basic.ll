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

define i32 @test2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
; CHECK: entry = 1024
entry:
  %cond = icmp ult i32 %i, 42
  br i1 %cond, label %then, label %else, !prof !0

; The 'then' branch is predicted more likely via branch weight metadata.
; CHECK: then = 963
then:
  br label %exit

; CHECK: else = 60
else:
  br label %exit

; FIXME: It may be a bug that we don't sum back to 1024.
; CHECK: exit = 1023
exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

!0 = metadata !{metadata !"branch_weights", i32 64, i32 4}

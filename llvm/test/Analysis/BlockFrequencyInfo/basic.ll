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

define i32 @test3(i32 %i, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK: Printing analysis {{.*}} for function 'test3'
; CHECK: entry = 1024
entry:
  switch i32 %i, label %case_a [ i32 1, label %case_b
                                 i32 2, label %case_c
                                 i32 3, label %case_d
                                 i32 4, label %case_e ], !prof !1

; CHECK: case_a = 51
case_a:
  br label %exit

; CHECK: case_b = 51
case_b:
  br label %exit

; The 'case_c' branch is predicted more likely via branch weight metadata.
; CHECK: case_c = 819
case_c:
  br label %exit

; CHECK: case_d = 51
case_d:
  br label %exit

; CHECK: case_e = 51
case_e:
  br label %exit

; FIXME: It may be a bug that we don't sum back to 1024.
; CHECK: exit = 1023
exit:
  %result = phi i32 [ %a, %case_a ],
                    [ %b, %case_b ],
                    [ %c, %case_c ],
                    [ %d, %case_d ],
                    [ %e, %case_e ]
  ret i32 %result
}

!1 = metadata !{metadata !"branch_weights", i32 4, i32 4, i32 64, i32 4, i32 4}

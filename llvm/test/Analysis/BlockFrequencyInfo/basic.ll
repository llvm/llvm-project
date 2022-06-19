; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

define i32 @test1(i32 %i, i32* %a) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test1':
; CHECK-NEXT: block-frequency-info: test1
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %body

; Loop backedges are weighted and thus their bodies have a greater frequency.
; CHECK-NEXT: body: float = 32.0,
body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i32 %iv
  %0 = load i32, i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  ret i32 %sum
}

define i32 @test2(i32 %i, i32 %a, i32 %b) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test2':
; CHECK-NEXT: block-frequency-info: test2
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  %cond = icmp ult i32 %i, 42
  br i1 %cond, label %then, label %else, !prof !0

; The 'then' branch is predicted more likely via branch weight metadata.
; CHECK-NEXT: then: float = 0.9411{{[0-9]*}},
then:
  br label %exit

; CHECK-NEXT: else: float = 0.05882{{[0-9]*}},
else:
  br label %exit

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

!0 = !{!"branch_weights", i32 64, i32 4}

define i32 @test3(i32 %i, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test3':
; CHECK-NEXT: block-frequency-info: test3
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  switch i32 %i, label %case_a [ i32 1, label %case_b
                                 i32 2, label %case_c
                                 i32 3, label %case_d
                                 i32 4, label %case_e ], !prof !1

; CHECK-NEXT: case_a: float = 0.05,
case_a:
  br label %exit

; CHECK-NEXT: case_b: float = 0.05,
case_b:
  br label %exit

; The 'case_c' branch is predicted more likely via branch weight metadata.
; CHECK-NEXT: case_c: float = 0.8,
case_c:
  br label %exit

; CHECK-NEXT: case_d: float = 0.05,
case_d:
  br label %exit

; CHECK-NEXT: case_e: float = 0.05,
case_e:
  br label %exit

; CHECK-NEXT: exit: float = 1.0, int = [[ENTRY]]
exit:
  %result = phi i32 [ %a, %case_a ],
                    [ %b, %case_b ],
                    [ %c, %case_c ],
                    [ %d, %case_d ],
                    [ %e, %case_e ]
  ret i32 %result
}

!1 = !{!"branch_weights", i32 4, i32 4, i32 64, i32 4, i32 4}

define void @nested_loops(i32 %a) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'nested_loops':
; CHECK-NEXT: block-frequency-info: nested_loops
; CHECK-NEXT: entry: float = 1.0, int = [[ENTRY:[0-9]+]]
entry:
  br label %for.cond1.preheader

; CHECK-NEXT: for.cond1.preheader: float = 4001.0,
for.cond1.preheader:
  %x.024 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  br label %for.cond4.preheader

; CHECK-NEXT: for.cond4.preheader: float = 16007984.8,
for.cond4.preheader:
  %y.023 = phi i32 [ 0, %for.cond1.preheader ], [ %inc9, %for.inc8 ]
  %add = add i32 %y.023, %x.024
  br label %for.body6

; CHECK-NEXT: for.body6: float = 64047914563.9,
for.body6:
  %z.022 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.body6 ]
  %add7 = add i32 %add, %z.022
  tail call void @g(i32 %add7)
  %inc = add i32 %z.022, 1
  %cmp5 = icmp ugt i32 %inc, %a
  br i1 %cmp5, label %for.inc8, label %for.body6, !prof !2

; CHECK-NEXT: for.inc8: float = 16007984.8,
for.inc8:
  %inc9 = add i32 %y.023, 1
  %cmp2 = icmp ugt i32 %inc9, %a
  br i1 %cmp2, label %for.inc11, label %for.cond4.preheader, !prof !2

; CHECK-NEXT: for.inc11: float = 4001.0,
for.inc11:
  %inc12 = add i32 %x.024, 1
  %cmp = icmp ugt i32 %inc12, %a
  br i1 %cmp, label %for.end13, label %for.cond1.preheader, !prof !2

; CHECK-NEXT: for.end13: float = 1.0, int = [[ENTRY]]
for.end13:
  ret void
}

declare void @g(i32)

!2 = !{!"branch_weights", i32 1, i32 4000}

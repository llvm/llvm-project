; RUN: opt -debug-only=branch-prob -passes=jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we set the branch probability for the newly created
; basic block.

define void @foo(i1 %cond1, i1 %cond2) !prof !0 !PGOFuncName !1 {
entry:
  br i1 %cond1, label %bb.f1, label %bb.f2, !prof !2

bb.f1:
  call void @f1()
  br label %bb.cond2

bb.f2:
  call void @f2()
  br label %bb.cond2

bb.cond2:
  br i1 %cond2, label %exit, label %bb.cond1again, !prof !3
; CHECK: set edge bb.cond2.thread -> 0 successor probability to 0x79b9d244 / 0x80000000
; CHECK: set edge bb.cond2.thread -> 1 successor probability to 0x06462dbc / 0x80000000 = 4.90

bb.cond1again:
  br i1 %cond1, label %bb.f3, label %bb.f4, !prof !4

bb.f3:
  call void @f3()
  br label %exit

bb.f4:
  call void @f4()
  br label %exit

exit:
  ret void
}

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()

!0 = !{!"function_entry_count", i64 15985}
!1 = !{!"foo.cpp:foo"}
!2 = !{!"branch_weights", i32 0, i32 36865}
!3 = !{!"branch_weights", i32 35058, i32 1807}
!4 = !{!"branch_weights", i32 1807, i32 35058}

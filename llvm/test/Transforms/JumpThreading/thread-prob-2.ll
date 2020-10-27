; RUN: opt -debug-only=branch-prob -jump-threading -S %s 2>&1 | FileCheck %s

; Make sure that we set the branch probability for the newly created
; basic block.

define void @foo(i32 %v0, i1 %arg2) !prof !0 !PGOFuncName !1 {
entry:
  %bool1 = icmp eq i32 %v0, 0
  br i1 %bool1, label %bb2, label %bb1, !prof !2

bb1:
  %sel = select i1 %arg2, i32 %v0, i32 0, !prof !3
  br label %bb2
; CHECK: set edge select.unfold -> 0 successor probability to 0x80000000 / 0x80000000

bb2:
  %phi = phi i32 [ %sel, %bb1 ], [ 0, %entry ]
  %bool2 = icmp eq i32 %phi, 0
  br i1 %bool2, label %exit, label %bb3, !prof !4

bb3:
  br label %exit

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 15985}
!1 = !{!"foo.cpp:foo"}
!2 = !{!"branch_weights", i32 0, i32 36865}
!3 = !{!"branch_weights", i32 35058, i32 1807}
!4 = !{!"branch_weights", i32 1807, i32 35058}

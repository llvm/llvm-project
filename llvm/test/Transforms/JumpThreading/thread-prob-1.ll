; RUN: opt -debug-only=branch-prob -jump-threading -S %s 2>&1 | FileCheck %s

; Make sure that we set the branch probability for the newly created
; basic block.

define void @foo(i1 %arg1, i1 %arg2, i32 %arg3) !prof !0 !PGOFuncName !1 {
entry:
  call void @bar(i32 0)
  br i1 %arg1, label %bb3, label %bb1, !prof !2

bb1:
  call void @bar(i32 1)
  br i1 %arg2, label %bb2, label %bb3, !prof !3

bb2:
  call void @bar(i32 2)
  br label %bb3

bb3:
; CHECK: set edge bb3.thr_comm -> 0 successor probability to 0x80000000 / 0x80000000
%ptr = phi i32 [ 0, %bb1 ], [ 0, %entry ], [ %arg3, %bb2 ]
  call void @bar(i32 3)
  %bool = icmp eq i32 %ptr, 0
  br i1 %bool, label %exit, label %bb4, !prof !4
; CHECK: set edge bb3.thread -> 0 successor probability to 0x80000000 / 0x80000000

bb4:
  call void @bar(i32 %ptr)
  br label %exit

exit:
  ret void
}

declare void @bar(i32)

!0 = !{!"function_entry_count", i64 15985}
!1 = !{!"foo:foo"}
!2 = !{!"branch_weights", i32 15973, i32 36865}
!3 = !{!"branch_weights", i32 2957, i32 5798}
!4 = !{!"branch_weights", i32 1807, i32 35058}
!5 = !{!"branch_weights", i32 38, i32 287958}

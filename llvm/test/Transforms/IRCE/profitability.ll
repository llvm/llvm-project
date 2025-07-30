; RUN: opt -S -verify-loop-info -irce-print-changed-loops -passes=irce -irce-min-eliminated-checks=51 < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-NO
; RUN: opt -S -verify-loop-info -irce-print-changed-loops -passes=irce -irce-min-eliminated-checks=50 < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-YES

; CHECK-YES: constrained Loop
; CHECK-NO-NOT: constrained Loop

declare void @bar(i32)

define i32 @foo(ptr %arr_a, ptr %a_len_ptr, i32 %n) {
entry:
  %len.a = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit, !prof !1

loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %backedge ]
  %abc.a = icmp slt i32 %idx, %len.a
  br i1 %abc.a, label %in.bounds.a, label %backedge, !prof !2

in.bounds.a:
  %addr.a = getelementptr i32, ptr %arr_a, i32 %idx
  %val = load i32, ptr %addr.a
  call void @bar(i32 %val)
  br label %backedge

backedge:
  %idx.next = add i32 %idx, 1
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit, !prof !3

exit:
  ret i32 0
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 1024, i32 1}
!2 = !{!"branch_weights", i32 1, i32 1}
!3 = !{!"branch_weights", i32 99, i32 1}

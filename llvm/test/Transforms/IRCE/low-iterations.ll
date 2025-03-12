; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -irce-min-eliminated-checks=3 < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-NO
; RUN: opt -verify-loop-info -irce-print-changed-loops -passes=irce -irce-min-eliminated-checks=0 < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-YES

; CHECK-YES: constrained Loop
; CHECK-NO-NOT: constrained Loop

define i32 @multiple_access_no_preloop(
  ptr %arr_a, ptr %a_len_ptr, ptr %arr_b, ptr %b_len_ptr, i32 %n) {

  entry:
  %len.a = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit, !prof !1

  loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %backedge ]
  %idx.next = add i32 %idx, 1
  %abc.a = icmp slt i32 %idx, %len.a
  br i1 %abc.a, label %in.bounds.a, label %exit, !prof !2

  in.bounds.a:
  %addr.a = getelementptr i32, ptr %arr_a, i32 %idx
  %val = load i32, ptr %addr.a
  %cond = icmp ne i32 %val, 0
; Most probable exit from a loop.
  br i1 %cond, label %found, label %backedge, !prof !3

  backedge:
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit, !prof !4

  found:
  ret i32 %val

  exit:
  ret i32 0
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 1024, i32 1}
!2 = !{!"branch_weights", i32 512, i32 1}
!3 = !{!"branch_weights", i32 1, i32 2}
!4 = !{!"branch_weights", i32 512, i32 1}

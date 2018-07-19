; RUN: opt -verify-loop-info -irce-print-changed-loops -irce < %s 2>&1 | FileCheck %s

; CHECK-NOT: constrained Loop

define void @multiple_access_no_preloop(
    i32* %arr_a, i32* %a_len_ptr, i32* %arr_b, i32* %b_len_ptr, i32 %n) {

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.b = load i32, i32* %b_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds.b ]
  %idx.next = add i32 %idx, 1
  %abc.a = icmp slt i32 %idx, %len.a
  br i1 %abc.a, label %in.bounds.a, label %out.of.bounds, !prof !1

 in.bounds.a:
  %addr.a = getelementptr i32, i32* %arr_a, i32 %idx
  store i32 0, i32* %addr.a
  %abc.b = icmp slt i32 %idx, %len.b
  br i1 %abc.b, label %in.bounds.b, label %out.of.bounds, !prof !1

 in.bounds.b:
  %addr.b = getelementptr i32, i32* %arr_b, i32 %idx
  store i32 -1, i32* %addr.b
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 1, i32 1}
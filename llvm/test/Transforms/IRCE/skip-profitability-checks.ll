; RUN: opt -irce-skip-profitability-checks -S -verify-loop-info -irce < %s | FileCheck %s
; RUN: opt -irce-skip-profitability-checks -S -verify-loop-info -passes='require<branch-prob>,irce' < %s | FileCheck %s

define void @single_access_no_preloop_no_offset(ptr %arr, ptr %a_len_ptr, i32 %n) {
; CHECK-LABEL: @single_access_no_preloop_no_offset(
; CHECK: main.exit.selector:
 entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, ptr %arr, i32 %idx
  store i32 0, ptr %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 1, i32 64}

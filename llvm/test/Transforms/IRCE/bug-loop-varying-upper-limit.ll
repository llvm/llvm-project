; RUN: opt -irce-print-changed-loops -S -verify-loop-info -irce -verify < %s 2>&1 | FileCheck %s
; RUN: opt -irce-print-changed-loops -S -verify-loop-info -passes='require<branch-prob>,loop(irce)' -verify < %s 2>&1 | FileCheck %s

; CHECK-NOT: constrained loop

define void @single_access_no_preloop_no_offset(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %len = load i32, i32* %a_len_ptr, !range !0
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}

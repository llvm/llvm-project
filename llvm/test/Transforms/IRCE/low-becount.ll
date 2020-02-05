; RUN: opt -irce-print-changed-loops -verify-loop-info -irce -S < %s 2>&1 | FileCheck %s
;
; TODO: new-pm version should be enabled after we decide on branch-probability handling for loop passes
; TODO: opt -irce-print-changed-loops -verify-loop-info -passes='require<branch-prob>,irce' -S < %s 2>&1 | FileCheck %s

; CHECK-NOT: constrained Loop

define void @low_profiled_be_count(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit, !prof !2

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
!2 = !{!"branch_weights", i32 4, i32 64}

; RUN: opt -verify-loop-info -irce -irce-print-range-checks -irce-print-changed-loops %s -S 2>&1 | FileCheck %s
; RUN: opt -verify-loop-info -passes='require<branch-prob>,irce' -irce-print-range-checks -irce-print-changed-loops %s -S 2>&1 | FileCheck %s

; Make sure that we can pick up both range checks.
define void @test_01(ptr %arr, ptr %a_len_ptr, ptr %size_ptr) {

; CHECK-LABEL: @test_01(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %size = load i32, ptr %size_ptr
  %first_iter_check = icmp sle i32 %size, 0
  br i1 %first_iter_check, label %exit, label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp slt i32 %iv, %len
  %rc2 = icmp slt i32 %iv, %size
  ; CHECK: %rc = and i1 true, true
  %rc = and i1 %rc1, %rc2
  br i1 %rc, label %backedge, label %out_of_bounds


backedge:
  %iv.next = add i32 %iv, 1
  %arr_el_ptr = getelementptr i32, ptr %arr, i32 %iv
  %el = load i32, ptr %arr_el_ptr
  %loopcond = icmp ne i32 %iv, %size
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void

out_of_bounds:
  ret void
}

; Same as test_01, unsigned predicates.
define void @test_02(ptr %arr, ptr %a_len_ptr, ptr %size_ptr) {

; CHECK-LABEL: @test_02(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %size = load i32, ptr %size_ptr
  %first_iter_check = icmp sle i32 %size, 0
  br i1 %first_iter_check, label %exit, label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp ult i32 %iv, %len
  %rc2 = icmp ult i32 %iv, %size
  ; CHECK: %rc = and i1 true, true
  %rc = and i1 %rc1, %rc2
  br i1 %rc, label %backedge, label %out_of_bounds


backedge:
  %iv.next = add i32 %iv, 1
  %arr_el_ptr = getelementptr i32, ptr %arr, i32 %iv
  %el = load i32, ptr %arr_el_ptr
  %loopcond = icmp ne i32 %iv, %size
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void

out_of_bounds:
  ret void
}

define void @test_03(ptr %arr, ptr %a_len_ptr, ptr %size_ptr) {

; CHECK-LABEL: @test_03(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %size = load i32, ptr %size_ptr
  %first_iter_check = icmp eq i32 %size, 0
  br i1 %first_iter_check, label %exit, label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp slt i32 %iv, %len
  %rc2 = icmp slt i32 %iv, %size
  ; CHECK: %rc = and i1 true, true
  %rc = and i1 %rc1, %rc2
  br i1 %rc, label %backedge, label %out_of_bounds


backedge:
  %iv.next = add i32 %iv, 1
  %arr_el_ptr = getelementptr i32, ptr %arr, i32 %iv
  %el = load i32, ptr %arr_el_ptr
  %loopcond = icmp ne i32 %iv, %len
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void

out_of_bounds:
  ret void
}

define void @test_04(ptr %arr, ptr %a_len_ptr, ptr %size_ptr) {

; CHECK-LABEL: @test_04(

entry:
  %len = load i32, ptr %a_len_ptr, !range !0
  %size = load i32, ptr %size_ptr
  %first_iter_check = icmp eq i32 %size, 0
  br i1 %first_iter_check, label %exit, label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %backedge ]
  %rc1 = icmp ult i32 %iv, %len
  %rc2 = icmp ult i32 %iv, %size
  ; CHECK: %rc = and i1 true, true
  %rc = and i1 %rc1, %rc2
  br i1 %rc, label %backedge, label %out_of_bounds


backedge:
  %iv.next = add i32 %iv, 1
  %arr_el_ptr = getelementptr i32, ptr %arr, i32 %iv
  %el = load i32, ptr %arr_el_ptr
  %loopcond = icmp ne i32 %iv, %len
  br i1 %loopcond, label %loop, label %exit

exit:
  ret void

out_of_bounds:
  ret void
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}

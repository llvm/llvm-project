; RUN: opt -S -irce < %s | FileCheck %s

define void @f_0(i32 *%arr, i32 *%a_len_ptr, i32 %n, i1* %cond_buf) {
; CHECK-LABEL: @f_0(

; CHECK-LABEL: loop.preheader:
; CHECK: [[not_n:[^ ]+]] = sub i32 -1, %n
; CHECK: [[not_safe_range_end:[^ ]+]] = sub i32 3, %len
; CHECK: [[not_exit_main_loop_at_hiclamp_cmp:[^ ]+]] = icmp sgt i32 [[not_n]], [[not_safe_range_end]]
; CHECK: [[not_exit_main_loop_at_hiclamp:[^ ]+]] = select i1 [[not_exit_main_loop_at_hiclamp_cmp]], i32 [[not_n]], i32 [[not_safe_range_end]]
; CHECK: [[exit_main_loop_at_hiclamp:[^ ]+]] = sub i32 -1, [[not_exit_main_loop_at_hiclamp]]
; CHECK: [[exit_main_loop_at_loclamp_cmp:[^ ]+]] = icmp sgt i32 [[exit_main_loop_at_hiclamp]], 0
; CHECK: [[exit_main_loop_at_loclamp:[^ ]+]] = select i1 [[exit_main_loop_at_loclamp_cmp]], i32 [[exit_main_loop_at_hiclamp]], i32 0
; CHECK: [[enter_main_loop:[^ ]+]] = icmp slt i32 0, [[exit_main_loop_at_loclamp]]
; CHECK: br i1 [[enter_main_loop]], label %loop, label %main.pseudo.exit

 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.for.abc = add i32 %idx, 4
  %abc.actual = icmp slt i32 %idx.for.abc, %len
  %cond = load volatile i1, i1* %cond_buf
  %abc = and i1 %cond, %abc.actual
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

; CHECK: loop:
; CHECK:  %cond = load volatile i1, i1* %cond_buf
; CHECK:  %abc = and i1 %cond, true
; CHECK:  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx.for.abc
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

define void @f_1(
    i32* %arr_a, i32* %a_len_ptr, i32* %arr_b, i32* %b_len_ptr, i32 %n) {
; CHECK-LABEL: @f_1(

; CHECK-LABEL: loop.preheader:
; CHECK: [[not_len_b:[^ ]+]] = sub i32 -1, %len.b
; CHECK: [[not_len_a:[^ ]+]] = sub i32 -1, %len.a
; CHECK: [[smax_not_len_cond:[^ ]+]] = icmp sgt i32 [[not_len_b]], [[not_len_a]]
; CHECK: [[smax_not_len:[^ ]+]] = select i1 [[smax_not_len_cond]], i32 [[not_len_b]], i32 [[not_len_a]]
; CHECK: [[not_n:[^ ]+]] = sub i32 -1, %n
; CHECK: [[not_upper_limit_cond_loclamp:[^ ]+]] = icmp sgt i32 [[smax_not_len]], [[not_n]]
; CHECK: [[not_upper_limit_loclamp:[^ ]+]] = select i1 [[not_upper_limit_cond_loclamp]], i32 [[smax_not_len]], i32 [[not_n]]
; CHECK: [[upper_limit_loclamp:[^ ]+]] = sub i32 -1, [[not_upper_limit_loclamp]]
; CHECK: [[upper_limit_cmp:[^ ]+]] = icmp sgt i32 [[upper_limit_loclamp]], 0
; CHECK: [[upper_limit:[^ ]+]] = select i1 [[upper_limit_cmp]], i32 [[upper_limit_loclamp]], i32 0

 entry:
  %len.a = load i32, i32* %a_len_ptr, !range !0
  %len.b = load i32, i32* %b_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc.a = icmp slt i32 %idx, %len.a
  %abc.b = icmp slt i32 %idx, %len.b
  %abc = and i1 %abc.a, %abc.b
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

; CHECK: loop:
; CHECK:   %abc = and i1 true, true
; CHECK:   br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr.a = getelementptr i32, i32* %arr_a, i32 %idx
  store i32 0, i32* %addr.a
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
!1 = !{!"branch_weights", i32 64, i32 4}

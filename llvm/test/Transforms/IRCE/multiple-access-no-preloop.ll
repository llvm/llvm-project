; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s
; RUN: opt -verify-loop-info -passes='require<branch-prob>,irce' -S < %s | FileCheck %s

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

; CHECK-LABEL: @multiple_access_no_preloop(

; CHECK: loop.preheader:
; CHECK: [[smax_len_cond:[^ ]+]] = icmp slt i32 %len.b, %len.a
; CHECK: [[smax_len:[^ ]+]] = select i1 [[smax_len_cond]], i32 %len.b, i32 %len.a
; CHECK: [[upper_limit_cond_loclamp:[^ ]+]] = icmp slt i32 [[smax_len]], %n
; CHECK: [[upper_limit_loclamp:[^ ]+]] = select i1 [[upper_limit_cond_loclamp]], i32 [[smax_len]], i32 %n
; CHECK: [[upper_limit_cmp:[^ ]+]] = icmp sgt i32 [[upper_limit_loclamp]], 0
; CHECK: [[upper_limit:[^ ]+]] = select i1 [[upper_limit_cmp]], i32 [[upper_limit_loclamp]], i32 0

; CHECK: loop:
; CHECK: br i1 true, label %in.bounds.a, label %out.of.bounds

; CHECK: in.bounds.a:
; CHECK: br i1 true, label %in.bounds.b, label %out.of.bounds

; CHECK: in.bounds.b:
; CHECK: [[main_loop_cond:[^ ]+]] = icmp slt i32 %idx.next, [[upper_limit]]
; CHECK: br i1 [[main_loop_cond]], label %loop, label %main.exit.selector

; CHECK: in.bounds.b.postloop:
; CHECK: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 128, i32 4}

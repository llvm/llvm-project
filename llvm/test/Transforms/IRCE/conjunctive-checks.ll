; RUN: opt -S -verify-loop-info -irce < %s | FileCheck %s
; RUN: opt -S -verify-loop-info -passes='require<branch-prob>,irce' < %s | FileCheck %s

define void @f_0(i32 *%arr, i32 *%a_len_ptr, i32 %n, i1* %cond_buf) {
; CHECK-LABEL: @f_0(

; CHECK: loop.preheader:
; CHECK: [[len_sub:[^ ]+]] = add nsw i32 %len, -4
; CHECK: [[exit_main_loop_at_hiclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 %n, i32 [[len_sub]])
; CHECK: [[exit_main_loop_at_loclamp:[^ ]+]] = call i32 @llvm.smax.i32(i32 [[exit_main_loop_at_hiclamp]], i32 0)
; CHECK: [[enter_main_loop:[^ ]+]] = icmp slt i32 0, [[exit_main_loop_at_loclamp]]
; CHECK: br i1 [[enter_main_loop]], label %[[loop_preheader2:[^ ,]+]], label %main.pseudo.exit

; CHECK: [[loop_preheader2]]:
; CHECK: br label %loop

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
; CHECK:  br i1 %abc, label %in.bounds, label %[[loop_exit:[^ ,]+]], !prof !1

; CHECK: [[loop_exit]]:
; CHECK:  br label %out.of.bounds

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

; CHECK: loop.preheader:
; CHECK: [[smax_len:[^ ]+]] = call i32 @llvm.smin.i32(i32 %len.b, i32 %len.a)
; CHECK: [[upper_limit_loclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 [[smax_len]], i32 %n)
; CHECK: [[upper_limit:[^ ]+]] = call i32 @llvm.smax.i32(i32 [[upper_limit_loclamp]], i32 0)

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
; CHECK:   br i1 %abc, label %in.bounds, label %[[oob_loopexit:[^ ,]+]], !prof !1

; CHECK: [[oob_loopexit]]:
; CHECK-NEXT:  br label %out.of.bounds


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

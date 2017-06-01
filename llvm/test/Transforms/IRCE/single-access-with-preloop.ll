; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s

define void @single_access_with_preloop(i32 *%arr, i32 *%a_len_ptr, i32 %n, i32 %offset) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %array.idx = add i32 %idx, %offset
  %abc.high = icmp slt i32 %array.idx, %len
  %abc.low = icmp sge i32 %array.idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %array.idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: @single_access_with_preloop(
; CHECK: loop.preheader:
; CHECK: [[not_safe_start:[^ ]+]] = add i32 %offset, -1
; CHECK: [[not_n:[^ ]+]] = sub i32 -1, %n
; CHECK: [[not_exit_preloop_at_cond_loclamp:[^ ]+]] = icmp sgt i32 [[not_safe_start]], [[not_n]]
; CHECK: [[not_exit_preloop_at_loclamp:[^ ]+]] = select i1 [[not_exit_preloop_at_cond_loclamp]], i32 [[not_safe_start]], i32 [[not_n]]
; CHECK: [[exit_preloop_at_loclamp:[^ ]+]] = sub i32 -1, [[not_exit_preloop_at_loclamp]]
; CHECK: [[exit_preloop_at_cond:[^ ]+]] = icmp sgt i32 [[exit_preloop_at_loclamp]], 0
; CHECK: [[exit_preloop_at:[^ ]+]] = select i1 [[exit_preloop_at_cond]], i32 [[exit_preloop_at_loclamp]], i32 0


; CHECK: [[not_safe_start_2:[^ ]+]] = add i32 %offset, -1
; CHECK: [[not_safe_end:[^ ]+]] = sub i32 [[not_safe_start_2]], %len
; CHECK: [[not_exit_mainloop_at_cond_loclamp:[^ ]+]] = icmp sgt i32 [[not_safe_end]], [[not_n]]
; CHECK: [[not_exit_mainloop_at_loclamp:[^ ]+]] = select i1 [[not_exit_mainloop_at_cond_loclamp]], i32 [[not_safe_end]], i32 [[not_n]]
; CHECK: [[exit_mainloop_at_loclamp:[^ ]+]] = sub i32 -1, [[not_exit_mainloop_at_loclamp]]
; CHECK: [[exit_mainloop_at_cmp:[^ ]+]] = icmp sgt i32 [[exit_mainloop_at_loclamp]], 0
; CHECK: [[exit_mainloop_at:[^ ]+]] = select i1 [[exit_mainloop_at_cmp]], i32 [[exit_mainloop_at_loclamp]], i32 0


; CHECK: in.bounds:
; CHECK: [[continue_mainloop_cond:[^ ]+]] = icmp slt i32 %idx.next, [[exit_mainloop_at]]
; CHECK: br i1 [[continue_mainloop_cond]], label %loop, label %main.exit.selector

; CHECK: main.exit.selector:
; CHECK: [[mainloop_its_left:[^ ]+]] = icmp slt i32 %idx.next.lcssa, %n
; CHECK: br i1 [[mainloop_its_left]], label %main.pseudo.exit, label %exit.loopexit

; CHECK: in.bounds.preloop:
; CHECK: [[continue_preloop_cond:[^ ]+]] = icmp slt i32 %idx.next.preloop, [[exit_preloop_at]]
; CHECK: br i1 [[continue_preloop_cond]], label %loop.preloop, label %preloop.exit.selector

; CHECK: preloop.exit.selector:
; CHECK: [[preloop_its_left:[^ ]+]] = icmp slt i32 %idx.next.preloop.lcssa, %n
; CHECK: br i1 [[preloop_its_left]], label %preloop.pseudo.exit, label %exit.loopexit

; CHECK: in.bounds.postloop:
; CHECK: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}

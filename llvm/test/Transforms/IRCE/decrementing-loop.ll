; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s

define void @decrementing_loop(i32 *%arr, i32 *%a_len_ptr, i32 %n) {
 entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  %start = sub i32 %n, 1
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ %start, %entry ] , [ %idx.dec, %in.bounds ]
  %idx.dec = sub i32 %idx, 1
  %abc.high = icmp slt i32 %idx, %len
  %abc.low = icmp sge i32 %idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp sgt i32 %idx.dec, -1
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void

; CHECK: loop.preheader:
; CHECK:   [[not_len:[^ ]+]] = sub i32 -1, %len
; CHECK:   [[not_n:[^ ]+]] = sub i32 -1, %n
; CHECK:   [[not_len_hiclamp_cmp:[^ ]+]] = icmp sgt i32 [[not_len]], [[not_n]]
; CHECK:   [[not_len_hiclamp:[^ ]+]] = select i1 [[not_len_hiclamp_cmp]], i32 [[not_len]], i32 [[not_n]]
; CHECK:   [[len_hiclamp:[^ ]+]] = sub i32 -1, [[not_len_hiclamp]]
; CHECK:   [[not_exit_preloop_at_cmp:[^ ]+]] = icmp sgt i32 [[len_hiclamp]], 0
; CHECK:   [[not_exit_preloop_at:[^ ]+]] = select i1 [[not_exit_preloop_at_cmp]], i32 [[len_hiclamp]], i32 0
; CHECK:   %exit.preloop.at = add i32 [[not_exit_preloop_at]], -1
}

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}

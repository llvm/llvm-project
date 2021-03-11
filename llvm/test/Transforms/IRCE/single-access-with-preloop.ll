; RUN: opt -verify-loop-info -irce -S < %s | FileCheck %s
; RUN: opt -verify-loop-info -passes='require<branch-prob>,irce' -S < %s | FileCheck %s

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
; CHECK: [[safe_offset_preloop:[^ ]+]] = call i32 @llvm.smax.i32(i32 %offset, i32 -2147483647)
; If Offset was a SINT_MIN, we could have an overflow here. That is why we calculated its safe version.
; CHECK: [[safe_start:[^ ]+]] = sub i32 0, [[safe_offset_preloop]]
; CHECK: [[exit_preloop_at_loclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 %n, i32 [[safe_start]])
; CHECK: [[exit_preloop_at:[^ ]+]] = call i32 @llvm.smax.i32(i32 [[exit_preloop_at_loclamp]], i32 0)


; CHECK: [[len_minus_sint_max:[^ ]+]] = add nuw nsw i32 %len, -2147483647
; CHECK: [[safe_offset_mainloop:[^ ]+]] = call i32 @llvm.smax.i32(i32 %offset, i32 [[len_minus_sint_max]])
; If Offset was a SINT_MIN, we could have an overflow here. That is why we calculated its safe version.
; CHECK: [[safe_upper_end:[^ ]+]] = sub i32 %len, [[safe_offset_mainloop]]
; CHECK: [[exit_mainloop_at_loclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 %n, i32 [[safe_upper_end]])
; CHECK: [[safe_offset_mainloop_2:[^ ]+]] = call i32 @llvm.smax.i32(i32 %offset, i32 0)
; CHECK: [[safe_lower_end:[^ ]+]] = sub i32 2147483647, [[safe_offset_mainloop_2]]
; CHECK: [[exit_mainloop_at_hiclamp:[^ ]+]] = call i32 @llvm.smin.i32(i32 [[exit_mainloop_at_loclamp]], i32 [[safe_lower_end]])
; CHECK: [[exit_mainloop_at:[^ ]+]] = call i32 @llvm.smax.i32(i32 [[exit_mainloop_at_hiclamp]], i32 0)

; CHECK: mainloop:
; CHECK: br label %loop

; CHECK: loop:
; CHECK: %abc.high = icmp slt i32 %array.idx, %len
; CHECK: %abc.low = icmp sge i32 %array.idx, 0
; CHECK: %abc = and i1 true, true
; CHECK: br i1 %abc, label %in.bounds, label %[[loopexit:[^ ,]+]]

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

; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s
; Tests preRAsched support for VRegCycle interference.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

define void @t(i32 %src_width, ptr nocapture %src_copy_start, ptr nocapture %dst_copy_start, i32 %src_copy_start_index) nounwind optsize {
entry:
  %0 = icmp eq i32 %src_width, 0
  br i1 %0, label %return, label %bb

; Make sure the scheduler schedules all uses of the preincrement
; induction variable before defining the postincrement value.
; CHECK-LABEL: t:
; CHECK: %bb
; CHECK-NOT: mov
bb:                                               ; preds = %entry, %bb
  %j.05 = phi i32 [ %2, %bb ], [ 0, %entry ]
  %tmp = mul i32 %j.05, %src_copy_start_index
  %uglygep = getelementptr i8, ptr %src_copy_start, i32 %tmp
  %dst_copy_start_addr.03 = getelementptr float, ptr %dst_copy_start, i32 %j.05
  %1 = load float, ptr %uglygep, align 4
  store float %1, ptr %dst_copy_start_addr.03, align 4
  %2 = add i32 %j.05, 1
  %exitcond = icmp eq i32 %2, %src_width
  br i1 %exitcond, label %return, label %bb

return:                                           ; preds = %bb, %entry
  ret void
}

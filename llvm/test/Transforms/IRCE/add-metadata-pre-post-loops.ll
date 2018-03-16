; RUN: opt -irce -S < %s 2>&1 | FileCheck %s
; RUN: opt -passes='require<branch-prob>,loop(irce)' -S < %s 2>&1 | FileCheck %s

; test that the pre and post loops have loop metadata which disables any further
; loop optimizations.

; generates a post loop, which should have metadata !llvm.loop !2
; Function Attrs: alwaysinline
define void @inner_loop(i32* %arr, i32* %a_len_ptr, i32 %n) #0 {
; CHECK-LABEL: inner_loop(
; CHECK-LABEL: in.bounds.postloop
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit.loopexit, !llvm.loop !2, !irce.loop.clone !7

entry:
  %len = load i32, i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

loop:                                             ; preds = %in.bounds, %entry
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %abc = icmp slt i32 %idx, %len
  br i1 %abc, label %in.bounds, label %out.of.bounds, !prof !1

in.bounds:                                        ; preds = %loop
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

out.of.bounds:                                    ; preds = %loop
  ret void

exit:                                             ; preds = %in.bounds, %entry
  ret void
}

; add loop metadata for pre and post loops
define void @single_access_with_preloop(i32 *%arr, i32 *%a_len_ptr, i32 %n, i32 %offset) {
; CHECK-LABEL: @single_access_with_preloop(
; CHECK-LABEL: in.bounds.preloop
; CHECK: br i1 [[COND:%[^ ]+]], label %loop.preloop, label %preloop.exit.selector, !llvm.loop !8, !irce.loop.clone !7
; CHECK-LABEL: in.bounds.postloop
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit.loopexit, !llvm.loop !9, !irce.loop.clone !7
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
attributes #0 = { alwaysinline }

!0 = !{i32 0, i32 2147483647}
!1 = !{!"branch_weights", i32 64, i32 4}
!2 = distinct !{!2, !3, !4, !5, !6}
!3 = !{!"llvm.loop.unroll.disable"}
!4 = !{!"llvm.loop.vectorize.enable", i1 false}
!5 = !{!"llvm.loop.licm_versioning.disable"}
!6 = !{!"llvm.loop.distribute.enable", i1 false}
!7 = !{}
!8 = distinct !{!8, !3, !4, !5}
!9 = distinct !{!9, !3, !4, !5}

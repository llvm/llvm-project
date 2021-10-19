; RUN: opt < %s -loop-rotate -verify-memoryssa -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Ensure that "llvm.loop.vectorize.enable" metadata was not lost after loop-rotate.
; In past LoopRotate was clearing that metadata.
;
; See http://reviews.llvm.org/D3348 for details.

; CHECK-LABEL: @foo
; CHECK: loop_cond:
; CHECK-NOT: loop-inc
; CHECK: br i1 %cmp, label %return, label %loop_cond, !llvm.loop !0
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.vectorize.enable", i1 true}
define i32 @foo(i32 %a) {
entry:
  br label %loop_cond

loop_cond:
  %indx = phi i32 [ 1, %entry ], [ %inc, %loop_inc ]
  %cmp = icmp ne i32 %indx, %a
  br i1 %cmp, label %return, label %loop_inc

loop_inc:
  %inc = add i32 %indx, 1
  br label %loop_cond, !llvm.loop !0

return:
  ret i32 0
}

!0 = !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}

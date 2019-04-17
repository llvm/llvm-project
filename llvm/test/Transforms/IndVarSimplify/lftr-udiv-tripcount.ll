; RUN: opt < %s -indvars -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; It is okay to do LFTR on this loop even though the trip count is a
; division because in this case the division can be optimized to a
; shift.

define void @foo(i8* %a, i8 %n) nounwind uwtable ssp {
; CHECK-LABEL: @foo(
 entry:
  %e = icmp sgt i8 %n, 3
  br i1 %e, label %loop, label %exit

 loop:
; CHECK-LABEL: loop:
  %i = phi i8 [ 0, %entry ], [ %i.inc, %loop ]
  %i1 = phi i8 [ 0, %entry ], [ %i1.inc, %loop ]
  %i.inc = add nsw i8 %i, 4
  %i1.inc = add i8 %i1, 1
  store volatile i8 0, i8* %a
  %c = icmp slt i8 %i, %n
; CHECK-LABEL:  %exitcond = icmp ne i8 %i1.inc
  br i1 %c, label %loop, label %exit

 exit:
; CHECK-LABEL: exit:
  ret void
}

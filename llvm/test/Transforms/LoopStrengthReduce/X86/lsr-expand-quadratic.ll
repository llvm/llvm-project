; REQUIRES: x86-registered-target
; RUN: opt -loop-reduce -S < %s | FileCheck %s

; Strength reduction analysis here relies on IV Users analysis, that
; only finds users among instructions with types that are treated as
; legal by the data layout. When running this test on pure non-x86
; configs (for example, ARM 64), it gets confused with the target
; triple and uses a default data layout instead. This default layout
; does not have any legal types (even i32), so the transformation
; does not happen.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; PR15470: LSR miscompile. The test2 function should return '1'.
;
; SCEV expander cannot expand quadratic recurrences outside of the
; loop. This recurrence depends on %sub.us, so can't be expanded.
; We cannot fold SCEVUnknown (sub.us) with recurrences since it is
; declared after the loop.
;
; CHECK-LABEL: @test2
; CHECK-LABEL: test2.loop:
; CHECK:  %lsr.iv1 = phi i32 [ %lsr.iv.next2, %test2.loop ], [ -16777216, %entry ]
; CHECK:  %lsr.iv = phi i32 [ %lsr.iv.next, %test2.loop ], [ -1, %entry ]
; CHECK:  %lsr.iv.next = add nsw i32 %lsr.iv, 1
; CHECK:  %lsr.iv.next2 = add nsw i32 %lsr.iv1, 16777216
;
; CHECK-LABEL: for.end:
; CHECK:  %tobool.us = icmp eq i32 %lsr.iv.next2, 0
; CHECK:  %sub.us = select i1 %tobool.us, i32 0, i32 0
; CHECK:  %1 = sub i32 0, %sub.us
; CHECK:  %2 = add i32 %1, %lsr.iv.next
; CHECK:  %sext.us = mul i32 %lsr.iv.next2, %2
; CHECK:  %f = ashr i32 %sext.us, 24
; CHECK: ret i32 %f
define i32 @test2() {
entry:
  br label %test2.loop

test2.loop:
  %inc1115.us = phi i32 [ 0, %entry ], [ %inc11.us, %test2.loop ]
  %inc11.us = add nsw i32 %inc1115.us, 1
  %cmp.us = icmp slt i32 %inc11.us, 2
  br i1 %cmp.us, label %test2.loop, label %for.end

for.end:
  %tobool.us = icmp eq i32 %inc1115.us, 0
  %sub.us = select i1 %tobool.us, i32 0, i32 0
  %mul.us = shl i32 %inc1115.us, 24
  %sub.cond.us = sub nsw i32 %inc1115.us, %sub.us
  %sext.us = mul i32 %mul.us, %sub.cond.us
  %f = ashr i32 %sext.us, 24
  br label %exit

exit:
  ret i32 %f
}

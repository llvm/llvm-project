; RUN: opt -passes=indvars < %s
; PR9424: Attempt to use a SCEVCouldNotCompute object!
; The inner loop computes the Step and Start of the outer loop.
; Call that Vexit. The outer End value is max(2,Vexit), because
; the condition "icmp %4 < 2" does not guard the outer loop.
; SCEV knows that Vexit has range [2,4], so End == Vexit == Start.
; Now we have ExactBECount == 0. However, MinStart == 2 and MaxEnd == 4.
; Since the stride is variable and may wrap, we cannot compute
; MaxBECount. SCEV should override MaxBECount with ExactBECount.

define void @bar() nounwind {
entry:
  %. = select i1 undef, i32 2, i32 1
  br label %"5.preheader"

"4":                                              ; preds = %"5.preheader", %"4"
  %0 = phi i32 [ 0, %"5.preheader" ], [ %1, %"4" ]
  %1 = add nsw i32 %0, 1
  %2 = icmp sgt i32 %., %1
  br i1 %2, label %"4", label %"9"

"9":                                              ; preds = %"4"
  %3 = add i32 %6, 1
  %4 = add i32 %3, %1
  %5 = icmp slt i32 %4, 2
  br i1 %5, label %"5.preheader", label %return

"5.preheader":                                    ; preds = %"9", %entry
  %6 = phi i32 [ 0, %entry ], [ %4, %"9" ]
  br label %"4"

return:                                           ; preds = %"9"
  ret void
}

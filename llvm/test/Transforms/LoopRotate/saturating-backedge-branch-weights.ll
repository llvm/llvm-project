; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;
; Covers the per-block balance floor in updateBranchWeights() that protects
; the saturating subtraction on LoopBackWeight:
;
;     LoopBackWeight = OrigLoopBackedgeWeight >= EnterWeight
;                        ? OrigLoopBackedgeWeight - EnterWeight : 0;
;
; Without the floor, an inconsistent profile where
;   EnterWeight = PreHeaderEntries - ExitWeight0  >  OrigLoopBackedgeWeight
; clamps LoopBackWeight to 0 but leaves the rotated body's incoming weight
; (EnterWeight + LoopBackWeight) larger than its outgoing weight, breaking
; per-block weight conservation. The floor lifts ExitWeight0 to
;   max(ExitWeight0, PreHeaderEntries - OrigLoopBackedgeWeight)
; (capped at OrigLoopExitWeight) so EnterWeight stays within
; OrigLoopBackedgeWeight and the saturating branch is never taken.
;
; CFG (weights shown as {taken, not-taken}; each block's incoming branch
; weight equals the sum of its outgoing branch weights):
;
;   entry  --(c0)--> ph         !prof { 999,  1 }   ; PreHeaderEntries = 999
;   ph     --------> header                          ; uncond, weight 999
;   header --(hcmp)--> exit1 / body   !prof { 990, 10 }
;   body   --(bcmp)--> exit2 / latch  !prof {  9,  1 }   ; second loop exit
;   latch  ---------> header                         ; uncond, weight 1
;
; OrigLoopExitWeight (990) > OrigLoopBackedgeWeight (10), so the
; "0-trip and 1-trip only" arm of the ExitWeight0 guess fires first:
;
;     ExitWeight0(initial) = OrigLoopExitWeight - OrigLoopBackedgeWeight
;                          = 990 - 10 = 980
;
; Floor (this fix): with PreHeaderEntries (999) > OrigBackedge (10),
;
;     MinExitWeight0 = PreHeaderEntries - OrigLoopBackedgeWeight
;                    = 999 - 10 = 989
;     ExitWeight0    = max(ExitWeight0, MinExitWeight0) = max(980, 989) = 989
;
; Remaining computation:
;
;     ExitWeight1    = OrigLoopExitWeight - ExitWeight0  = 990 - 989 = 1
;     EnterWeight    = PreHeaderEntries   - ExitWeight0  = 999 - 989 = 10
;     LoopBackWeight = OrigBackedge       - EnterWeight  = 10  - 10  = 0
;                      ; no longer saturated; EnterWeight == OrigBackedge.
;
;     GUARD  !prof { 989, 10 }
;     LATCH  !prof {   1,  0 }
;
; Per-block conservation after rotation:
;   ph    : in 999, out 989 + 10  = 999
;   .lr.ph: in 10,  out 10
;   body  : in 10 (.lr.ph) + 0 (latch) = 10, out 9 + 1 = 10
;   latch : in 1,   out 1 + 0 = 1
;   exit1 : in 989 (ph) + 1 (latch) = 990
;   exit2 : in 9
;   ret   : in 1
;   exits : 990 + 9 + 1 = 1000 = function_entry_count

define void @sat_backedge(ptr %p, i32 %n, i32 %cond) !prof !14 {
entry:
  %c0 = icmp ne i32 %cond, 0
  br i1 %c0, label %ph, label %ret, !prof !15

ph:                                               ; preds = %entry
  br label %header

header:                                           ; preds = %latch, %ph
  %iv = phi i32 [ 0, %ph ], [ %iv.next, %latch ]
  %hcmp = icmp sge i32 %iv, %n
  br i1 %hcmp, label %exit1, label %body, !prof !16

body:                                             ; preds = %header
  %addr = getelementptr i32, ptr %p, i32 %iv
  %v = load i32, ptr %addr, align 4
  %bcmp = icmp slt i32 %v, 0
  br i1 %bcmp, label %exit2, label %latch, !prof !17

latch:                                            ; preds = %body
  %iv.next = add i32 %iv, 1
  br label %header

exit1:                                            ; preds = %header
  ret void

exit2:                                            ; preds = %body
  ret void

ret:                                              ; preds = %entry
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 11000}
!4 = !{!"MaxCount", i64 990}
!5 = !{!"MaxInternalCount", i64 990}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 4}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 990, i32 1}
!12 = !{i32 999000, i64 1000, i32 1}
!13 = !{i32 999999, i64 990, i32 1}
!14 = !{!"function_entry_count", i64 1000}
!15 = !{!"branch_weights", i32 999, i32 1}
!16 = !{!"branch_weights", i32 990, i32 10}
!17 = !{!"branch_weights", i32 9,   i32 1}

; CHECK-LABEL: define void @sat_backedge(

; Guard at the new preheader: ExitWeight0 floored to
; PreHeaderEntries - OrigBackedge = 999 - 10 = 989; EnterWeight = 10.
; CHECK:      ph:
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %{{.*}}.lr.ph, !prof [[GUARD:![0-9]+]]

; Rotated latch: EnterWeight == OrigBackedge so LoopBackWeight = 0 without
; tripping the saturating subtraction; the rotated body's incoming weight
; (10) matches its outgoing weight (9 + 1).
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %body, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[GUARD]] = !{!"branch_weights", i32 989, i32 10}
; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 1,   i32 0}

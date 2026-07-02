; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;
; Covers the "loop never entered" case in updateBranchWeights()
; (OrigLoopBackedgeWeight == 0).
;
;   entry --(c0)--> ph        !prof {5, 1}   ; PreHeaderEntries = 5
;   ph    -------> header
;   header -> exit / latch    !prof {5, 0}   ; no backedge
;   latch  -------> header
;
; Expected rotated header: !prof {5, 0}.

define void @g(i32 %n, i32 %cond) !prof !14 {
entry:
  %c0 = icmp ne i32 %cond, 0
  br i1 %c0, label %ph, label %ret, !prof !15

ph:                                               ; preds = %entry
  br label %header

header:                                           ; preds = %latch, %ph
  %iv = phi i32 [ 0, %ph ], [ %iv.next, %latch ]
  %hcmp = icmp sge i32 %iv, %n
  br i1 %hcmp, label %exit, label %latch, !prof !16

latch:                                            ; preds = %header
  %iv.next = add i32 %iv, 1
  br label %header

exit:                                             ; preds = %header
  ret void

ret:                                              ; preds = %entry
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 6}
!4 = !{!"MaxCount", i64 5}
!5 = !{!"MaxInternalCount", i64 5}
!6 = !{!"MaxFunctionCount", i64 6}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 5, i32 1}
!12 = !{i32 999000, i64 5, i32 1}
!13 = !{i32 999999, i64 5, i32 1}
!14 = !{!"function_entry_count", i64 6}
!15 = !{!"branch_weights", i32 5, i32 1}
!16 = !{!"branch_weights", i32 5, i32 0}

; CHECK-LABEL: define void @g(

; CHECK:        br i1 %{{.*}}, label %exit, label %header, !prof [[GW:![0-9]+]]

; CHECK-DAG: [[GW]] = !{!"branch_weights", i32 5, i32 0}

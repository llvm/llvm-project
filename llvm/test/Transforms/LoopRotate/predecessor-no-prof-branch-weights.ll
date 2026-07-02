; RUN: opt < %s -passes=loop-rotate -S | FileCheck %s
;
; Covers the fallback in updateBranchWeights() when BFI is unavailable
; (plain loop-rotate run). PreHeaderEntries cannot be derived, so the
; single-exit formula is used:
;
;   EnterWeight    = ExitWeight1
;   LoopBackWeight = OrigLoopBackedgeWeight - EnterWeight
;
; Original CFG:
;   entry --(c0)--> ph
;   ph    -------> header
;   header -> exit / body   !prof {200, 9800}
;   body  -------> latch
;   latch -------> header
;
; Header weights are >= 127, so no rescaling occurs.
;
; Expected post-rotation:
;   ExitWeight0    = 1
;   ExitWeight1    = 199
;   EnterWeight    = 199
;   LoopBackWeight = 9601
;
;   GUARD !prof {1, 199}
;   LATCH !prof {199, 9601}

define void @no_bfi(ptr %p, i32 %n, i32 %cond) !prof !14 {
entry:
  %c0 = icmp ne i32 %cond, 0
  br i1 %c0, label %ph, label %ret

ph:                                               ; preds = %entry
  br label %header

header:                                           ; preds = %latch, %ph
  %iv = phi i32 [ 0, %ph ], [ %iv.next, %latch ]
  %hcmp = icmp sge i32 %iv, %n
  br i1 %hcmp, label %exit, label %body, !prof !15

body:                                             ; preds = %header
  %addr = getelementptr i32, ptr %p, i32 %iv
  store i32 %iv, ptr %addr, align 4
  br label %latch

latch:                                            ; preds = %body
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
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 9800}
!5 = !{!"MaxInternalCount", i64 9800}
!6 = !{!"MaxFunctionCount", i64 200}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 9800, i32 1}
!12 = !{i32 999000, i64 200, i32 1}
!13 = !{i32 999999, i64 200, i32 1}
!14 = !{!"function_entry_count", i64 201}
!15 = !{!"branch_weights", i32 200, i32 9800}

; CHECK-LABEL: define void @no_bfi(

; Preheader-bypass guard: derived from the old single-exit formula because
; BlockFrequencyInfo is not available to read PreHeaderEntries from.
; CHECK:      ph:
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %{{.*}}.lr.ph, !prof [[GUARD:![0-9]+]]

; Rotated latch test.
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %body, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[GUARD]] = !{!"branch_weights", i32 1,   i32 199}
; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 199, i32 9601}

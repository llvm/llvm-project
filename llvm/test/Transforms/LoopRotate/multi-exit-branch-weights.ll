; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;
; Multi-exit loop where preheader entry count differs from the header’s
; exit count.
;
;   entry --(c0)--> ph        !prof {210, 1}
;   ph    -------> header
;   header -> exit1 / body    !prof {200, 9800}
;   body   -> exit2 / latch   !prof {10, 9790}   ; second exit
;   latch  -------> header
;
; Entry counts are 10x the branch-weight scale. BFI reports
; PreHeaderEntries = 2100 and HeaderEntries = 100000, which must be
; scaled back to 210.
;
; Expected post-rotation:
;   ExitWeight0  + EnterWeight    = 210
;   ExitWeight0  + ExitWeight1    = 200
;   EnterWeight  + LoopBackWeight = 9800

define void @f(ptr %p, i32 %start, i32 %limit, i32 %cond) !prof !14 {
entry:
  %c0 = icmp ne i32 %cond, 0
  br i1 %c0, label %ph, label %ret, !prof !15

ph:                                               ; preds = %entry
  br label %header

header:                                           ; preds = %latch, %ph
  %iv = phi i32 [ %start, %ph ], [ %iv.next, %latch ]
  ; loop-rotate cannot prove %start != %limit, so it must materialize a guard.
  %hcmp = icmp eq i32 %iv, %limit
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
!3 = !{!"TotalCount", i64 108000}
!4 = !{!"MaxCount", i64 98000}
!5 = !{!"MaxInternalCount", i64 98000}
!6 = !{!"MaxFunctionCount", i64 2110}
!7 = !{!"NumCounts", i64 4}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 98000, i32 1}
!12 = !{i32 999000, i64 2110, i32 1}
!13 = !{i32 999999, i64 2000, i32 1}
!14 = !{!"function_entry_count", i64 2110}
!15 = !{!"branch_weights", i32 210, i32 1}
!16 = !{!"branch_weights", i32 200, i32 9800}
!17 = !{!"branch_weights", i32 10, i32 9790}

; CHECK-LABEL: define void @f(

; CHECK:      ph:
; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %{{.*}}.lr.ph, !prof [[GUARD:![0-9]+]]

; CHECK:        br i1 %{{.*}}, label %{{.*}}, label %body, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[GUARD]] = !{!"branch_weights", i32 1, i32 209}
; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 199, i32 9591}

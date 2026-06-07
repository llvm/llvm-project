; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;

; The cloned preheader branch folds to an unconditional loop entry since the
; initial IV cannot satisfy the exit condition. The loop remains multi-exit,
; so the rotated latch must use the BFI-derived first-iteration count instead
; of the single-exit estimate.
;
; Original profile scale:
;   entry  -> header                      ; PreHeaderEntries = 210
;   header -> exit1 / body  !prof {200, 9800}
;   body   -> exit2 / latch !prof {10, 9790}
;   latch  -> header
;
; Expected rotated latch (no guard, ExitWeight0 = 0):
;   ExitWeight1   = 200
;   EnterWeight   = 210
;   LoopBackWeight = 9800 - 210 = 9590
;   LATCH !prof {200, 9590}

define void @folded_preheader_multi_exit(ptr %p) !prof !14 {
entry:
  br label %header

header:                                           ; preds = %latch, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  %hcmp = icmp eq i32 %iv, 42
  br i1 %hcmp, label %exit1, label %body, !prof !15

body:                                             ; preds = %header
  %addr = getelementptr i32, ptr %p, i32 %iv
  %v = load i32, ptr %addr, align 4
  %bcmp = icmp slt i32 %v, 0
  br i1 %bcmp, label %exit2, label %latch, !prof !16

latch:                                            ; preds = %body
  %iv.next = add i32 %iv, 1
  br label %header

exit1:                                            ; preds = %header
  ret void

exit2:                                            ; preds = %body
  ret void

}

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10800}
!4 = !{!"MaxCount", i64 9800}
!5 = !{!"MaxInternalCount", i64 9800}
!6 = !{!"MaxFunctionCount", i64 210}
!7 = !{!"NumCounts", i64 4}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 9800, i32 1}
!12 = !{i32 999000, i64 210, i32 1}
!13 = !{i32 999999, i64 200, i32 1}
!14 = !{!"function_entry_count", i64 210}
!15 = !{!"branch_weights", i32 200, i32 9800}
!16 = !{!"branch_weights", i32 10, i32 9790}
; CHECK-LABEL: define void @folded_preheader_multi_exit(

; CHECK:      entry:
; CHECK-NEXT:   br label %body

; CHECK: br i1 %{{.*}}, label %{{.*}}, label %body, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 200, i32 9590}

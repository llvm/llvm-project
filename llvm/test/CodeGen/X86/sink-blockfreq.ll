; RUN: llc -disable-preheader-prot=true -disable-machine-licm -machine-sink-bfi=true -mtriple=x86_64-apple-darwin < %s | FileCheck %s -check-prefix=MSINK_BFI
; RUN: llc -disable-preheader-prot=true -disable-machine-licm -machine-sink-bfi=false -mtriple=x86_64-apple-darwin < %s | FileCheck %s -check-prefix=MSINK_NOBFI
; RUN: llc -disable-preheader-prot=true -disable-machine-licm -machine-sink-bfi=true -force-pgso -mtriple=x86_64-apple-darwin < %s | FileCheck %s -check-prefix=MSINK_NOBFI

; Test that by changing BlockFrequencyInfo we change the order in which
; machine-sink looks for successor blocks. By not using BFI, both G and B
; have the same loop depth and no instructions is sinked - B is selected but
; can't be used as to avoid breaking a non profitable critical edge. By using
; BFI, "mul" is sinked into the less frequent block G.
define i32 @sink_freqinfo(i32 %a, i32 %b) nounwind uwtable ssp !prof !14 {
; MSINK_BFI-LABEL: sink_freqinfo
; MSINK_BFI: jl
; MSINK_BFI-NEXT: ## %bb.
; MSINK_BFI-NEXT: imull

; MSINK_NOBFI-LABEL: sink_freqinfo
; MSINK_NOBFI: imull
; MSINK_NOBFI: jl
entry:
  br label %B

B:
  %ee = phi i32 [ 0, %entry ], [ %inc, %F ]
  %xx = sub i32 %a, %ee
  %cond0 = icmp slt i32 %xx, 0
  br i1 %cond0, label %F, label %exit, !prof !15

F:
  %inc = add nsw i32 %xx, 2
  %aa = mul nsw i32 %b, %inc
  %exitcond = icmp slt i32 %inc, %a
  br i1 %exitcond, label %B, label %G, !prof !16

G:
  %ii = add nsw i32 %aa, %a
  %ll = add i32 %b, 45
  %exitcond2 = icmp sge i32 %ii, %b
  br i1 %exitcond2, label %G, label %exit, !prof !17

exit:
  ret i32 0
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 1000}
!15 = !{!"branch_weights", i32 4, i32 1}
!16 = !{!"branch_weights", i32 128, i32 1}
!17 = !{!"branch_weights", i32 1, i32 1}

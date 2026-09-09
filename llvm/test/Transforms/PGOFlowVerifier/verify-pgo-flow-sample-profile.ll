; REQUIRES: asserts
; RUN: opt < %s -passes=instcombine -verify-pgo-flow \
; RUN:     -debug-only=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; SampleProfile summary is not InstrProf use-phase metadata.

; CHECK: *** PGO Flow Verification After InstCombinePass ***{{$}}
; CHECK: PGOFlowVerifier: skip 'f' (no InstrProf use-phase summary)

define i32 @f(i32 %x) {
  %a = add i32 %x, 0
  ret i32 %a
}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 1}
!5 = !{!"MaxCount", i64 1}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1}
!8 = !{!"NumCounts", i64 1}
!9 = !{!"NumFunctions", i64 1}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12}
!12 = !{i32 10000, i64 1, i32 1}

; RUN: opt < %s -passes='require<profile-summary>,function(chr)' -force-chr -chr-merge-threshold=1 -disable-output

; Regression test for a crash in CHR when setting unknown profdata on the
; merged condition. IRBuilder::CreateLogicalAnd is implemented as a select and
; can constant-fold to a non-Instruction value (e.g. `i1 true`). The buggy code
; assumed it always produced an Instruction and did `cast<Instruction>(V)`,
; which can segfault in release builds.
define void @repro_crash() {
entry:
  br i1 true, label %then, label %exit, !prof !15

then:
  br label %exit

exit:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 1}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11}
!11 = !{i32 999999, i64 1, i32 1}

!15 = !{!"branch_weights", i32 100, i32 1}

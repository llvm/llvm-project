; REQUIRES: x86-registered-target

; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions -O0 -mfs-psi-cutoff=0 -mfs-count-threshold=10000 | FileCheck %s

;; Check that functions with optnone attribute are not split.
; CHECK-LABEL: foo_optnone:
; CHECK-NOT:   .section .text.split.foo_optnone
; CHECK-NOT:   foo_optnone.cold:
; CHECK:       .LBB0_2:
; CHECK:       .size   foo_optnone

define void @foo_optnone(i1 zeroext %0) nounwind optnone noinline !prof !14 !section_prefix !15 {
entry:
  br i1 %0, label %hot, label %cold, !prof !17

hot:
  %1 = call i32 @bar()
  br label %exit

cold:
  %2 = call i32 @baz()
  br label %exit

exit:
  %3 = tail call i32 @qux()
  ret void
}

declare i32 @bar()
declare i32 @baz()
declare i32 @qux()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999900, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 7000}
!15 = !{!"function_section_prefix", !"hot"}
!17 = !{!"branch_weights", i32 7000, i32 0}

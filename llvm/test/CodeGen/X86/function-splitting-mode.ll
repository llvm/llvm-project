;; Tests for the -function-splitting= late function splitting mode.
; REQUIRES: x86-registered-target

;; The legacy spellings and -function-splitting=all are equivalent.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions | FileCheck %s --check-prefix=SPLIT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions | FileCheck %s --check-prefix=SPLIT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -function-splitting=all | FileCheck %s --check-prefix=SPLIT

;; A function which only has PGO data is not split unless the mode is 'all'.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=NOSPLIT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -function-splitting=none | FileCheck %s --check-prefix=NOSPLIT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -function-splitting=bbsections | FileCheck %s --check-prefix=NOSPLIT

;; An explicit mode overrides the legacy options.
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -split-machine-functions -function-splitting=none | FileCheck %s --check-prefix=NOSPLIT
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-split-machine-functions -function-splitting=bbsections | FileCheck %s --check-prefix=NOSPLIT

;; Invalid modes are rejected.
; RUN: not llc < %s -mtriple=x86_64-unknown-linux-gnu -function-splitting=bogus 2>&1 | FileCheck %s --check-prefix=ERR
; ERR: for the --function-splitting option: Cannot find option named 'bogus'!

define void @foo(i1 zeroext %0) nounwind !prof !14 !section_prefix !15 {
; SPLIT-LABEL:   foo
; SPLIT:         .section        .text.split.foo
; SPLIT-NEXT:    foo.cold:
; SPLIT-NOT:     callq   bar
; SPLIT-NEXT:    callq   baz
;
; NOSPLIT-LABEL: foo
; NOSPLIT-NOT:   .section        .text.split.foo
; NOSPLIT-NOT:   foo.cold:
  br i1 %0, label %2, label %4, !prof !17

2:                                                ; preds = %1
  %3 = call i32 @bar()
  br label %6

4:                                                ; preds = %1
  %5 = call i32 @baz()
  br label %6

6:                                                ; preds = %4, %2
  %7 = tail call i32 @qux()
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

; Modules compiled with different PGO profiles carry different ProfileSummary
; module flags, which cannot be linked. Importing across them must not fail:
; skip the conflicting source module instead.

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary app.ll -o app.bc
; RUN: opt -module-summary lib.ll -o lib.bc
; RUN: opt -module-summary same.ll -o same.bc
; RUN: llvm-lto -thinlto -o index app.bc lib.bc same.bc

; RUN: not llvm-link app.bc lib.bc -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LINK
; LINK: linking module flags 'ProfileSummary': IDs have conflicting values

;; lib.bc has a different summary and is skipped; same.bc has the same summary
;; and is imported from.
; RUN: opt -S -passes=function-import -summary-file index.thinlto.bc app.bc \
; RUN:   2>&1 | FileCheck %s
; CHECK-NOT: conflicting values
; CHECK: declare void @lib_profile()
; CHECK: define available_externally void @same_profile()
; CHECK-NOT: define available_externally void @lib_profile()

;--- app.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @main() {
  call void @lib_profile()
  call void @same_profile()
  ret void
}

declare void @lib_profile()
declare void @same_profile()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 1000}
!4 = !{!"MaxCount", i64 100}
!5 = !{!"MaxInternalCount", i64 10}
!6 = !{!"MaxFunctionCount", i64 100}
!7 = !{!"NumCounts", i64 10}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 4}

;--- lib.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @lib_profile() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 555}
!4 = !{!"MaxCount", i64 55}
!5 = !{!"MaxInternalCount", i64 5}
!6 = !{!"MaxFunctionCount", i64 55}
!7 = !{!"NumCounts", i64 5}
!8 = !{!"NumFunctions", i64 1}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 55, i32 1}
!12 = !{i32 999000, i64 55, i32 1}
!13 = !{i32 999999, i64 1, i32 4}

;--- same.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @same_profile() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 1000}
!4 = !{!"MaxCount", i64 100}
!5 = !{!"MaxInternalCount", i64 10}
!6 = !{!"MaxFunctionCount", i64 100}
!7 = !{!"NumCounts", i64 10}
!8 = !{!"NumFunctions", i64 3}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999000, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 4}

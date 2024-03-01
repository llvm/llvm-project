; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -stats 2>&1 \
; RUN:   -hwasan-skip-hot-code=1 -hwasan-percentile-cutoff-hot=700000 | FileCheck %s --check-prefix=PERCENT

; DEFAULT: 1 hwasan - Number of total funcs HWASAN

; PERCENT: 1 hwasan - Number of HWASAN instrumented funcs
; PERCENT: 1 hwasan - Number of total funcs HWASAN

define void @sanitized() sanitize_hwaddress !prof !36 { ret void }

!llvm.module.flags = !{!6}
!6 = !{i32 1, !"ProfileSummary", !7}
!7 = !{!8, !9, !10, !11, !12, !13, !14, !17}
!8 = !{!"ProfileFormat", !"InstrProf"}
!9 = !{!"TotalCount", i64 30000}
!10 = !{!"MaxCount", i64 10000}
!11 = !{!"MaxInternalCount", i64 10000}
!12 = !{!"MaxFunctionCount", i64 10000}
!13 = !{!"NumCounts", i64 3}
!14 = !{!"NumFunctions", i64 5}
!17 = !{!"DetailedSummary", !18}
!18 = !{!19, !29, !30, !32, !34}
!19 = !{i32 10000, i64 10000, i32 3}
!29 = !{i32 950000, i64 5000, i32 3}
!30 = !{i32 990000, i64 500, i32 4}
!32 = !{i32 999900, i64 250, i32 4}
!34 = !{i32 999999, i64 1, i32 6}

!36 = !{!"function_entry_count", i64 1000}

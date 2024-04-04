; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -hwasan-percentile-cutoff-hot=700000 | FileCheck %s --check-prefix=HOT70
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -hwasan-percentile-cutoff-hot=990000 | FileCheck %s --check-prefix=HOT99
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -hwasan-random-skip-rate=0.0 | FileCheck %s --check-prefix=RANDOM0
; RUN: opt < %s -passes='require<profile-summary>,hwasan' -S -hwasan-random-skip-rate=1.0 | FileCheck %s --check-prefix=RANDOM1

; HOT70: @sanitized
; HOT70-NEXT: @__hwasan_tls

; HOT99: @sanitized
; HOT99-NEXT: %x = alloca i8, i64 4

; RANDOM0: @sanitized
; RANDOM0-NEXT: @__hwasan_tls

; RANDOM1: @sanitized
; RANDOM1-NEXT: %x = alloca i8, i64 4

declare void @use(ptr)

define void @sanitized(i32 noundef %0) sanitize_hwaddress !prof !36 {
  %x = alloca i8, i64 4
  call void @use(ptr %x)
  ret void
}

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

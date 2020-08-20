; REQUIRES: x86-registered-target
; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=0 -codegenprepare -S < %s | FileCheck %s

; Test to ensure that split cold function gets 0 entry count profile
; metadata when compiling with pgo.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK: define {{.*}} @fun{{.*}} ![[HOTPROF:[0-9]+]] {{.*}}section_prefix ![[LIKELY:[0-9]+]]
; CHECK: call void @fun.cold.1

define void @fun() !prof !14 {
entry:
  br i1 undef, label %if.then, label %if.else

if.then:
  ret void

if.else:
  call void @sink()
  ret void
}

declare void @sink() cold

; CHECK: define {{.*}} @fun.cold.1{{.*}} ![[PROF:[0-9]+]] {{.*}}section_prefix ![[UNLIKELY:[0-9]+]]

; CHECK: ![[HOTPROF]] = !{!"function_entry_count", i64 100}
; CHECK: ![[LIKELY]] = !{!"function_section_prefix", !".hot"}
; CHECK: ![[PROF]] = !{!"function_entry_count", i64 0}
; CHECK: ![[UNLIKELY]] = !{!"function_section_prefix", !".unlikely"}

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
!14 = !{!"function_entry_count", i64 100}
!15 = !{!"function_section_prefix", !".hot"}
!16 = !{!"function_entry_count", i64 0}
!17 = !{!"function_section_prefix", !".unlikely"}

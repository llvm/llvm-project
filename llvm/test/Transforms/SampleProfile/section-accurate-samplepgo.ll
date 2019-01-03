; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline.prof -codegenprepare -S | FileCheck %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline.prof -codegenprepare -profile-sample-accurate -S | FileCheck %s --check-prefix ACCURATE

target triple = "x86_64-pc-linux-gnu"

; The test checks that function without profile gets unlikely section prefix
; if -profile-sample-accurate is specified or the function has the
; profile-sample-accurate attribute.

declare void @hot_func()

; CHECK-NOT: foo_not_in_profile{{.*}}!section_prefix
; CHECK: foo_not_in_profile{{.*}}!prof ![[UNKNOWN_ID:[0-9]+]]
; ACCURATE: foo_not_in_profile{{.*}}!prof ![[ZERO_ID:[0-9]+]] !section_prefix ![[COLD_ID:[0-9]+]]
; The function not appearing in profile is cold when -profile-sample-accurate
; is on.
define void @foo_not_in_profile() {
  call void @hot_func()
  ret void
}

; CHECK: bar_not_in_profile{{.*}}!prof ![[ZERO_ID:[0-9]+]] !section_prefix ![[COLD_ID:[0-9]+]]
; ACCURATE: bar_not_in_profile{{.*}}!prof ![[ZERO_ID:[0-9]+]] !section_prefix ![[COLD_ID:[0-9]+]]
; The function not appearing in profile is cold when the func has
; profile-sample-accurate attribute.
define void @bar_not_in_profile() #0 {
  call void @hot_func()
  ret void
}

attributes #0 = { "profile-sample-accurate" }

; CHECK: ![[UNKNOWN_ID]] = !{!"function_entry_count", i64 -1}
; CHECK: ![[ZERO_ID]] = !{!"function_entry_count", i64 0}
; CHECK: ![[COLD_ID]] = !{!"function_section_prefix", !".unlikely"}
; ACCURATE: ![[ZERO_ID]] = !{!"function_entry_count", i64 0}
; ACCURATE: ![[COLD_ID]] = !{!"function_section_prefix", !".unlikely"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}

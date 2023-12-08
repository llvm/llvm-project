; REQUIRES: x86-registered-target
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline.prof | opt -S -codegenprepare | FileCheck %s
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline.prof | opt -S -codegenprepare -profile-unknown-in-special-section -partial-profile | FileCheck %s --check-prefix=UNKNOWN
; RUN: opt -S %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline.prof -profile-sample-accurate -S | opt -S -codegenprepare | FileCheck %s --check-prefix=ACCURATE

target triple = "x86_64-pc-linux-gnu"

; The test checks that function without profile gets unlikely section prefix
; if -profile-sample-accurate is specified or the function has the
; profile-sample-accurate attribute.

declare void @hot_func()

; CHECK-NOT: foo_not_in_profile{{.*}}!section_prefix
; CHECK: foo_not_in_profile{{.*}}!prof ![[NOPROFILE_ID:[0-9]+]]
; UNKNOWN: foo_not_in_profile{{.*}}!prof ![[NOPROFILE_ID:[0-9]+]] !section_prefix ![[UNKNOWN_ID:[0-9]+]]
; ACCURATE: foo_not_in_profile{{.*}}!prof ![[ZERO_ID:[0-9]+]] !section_prefix ![[COLD_ID:[0-9]+]]
; The function not appearing in profile is cold when -profile-sample-accurate
; is on.
define void @foo_not_in_profile() #1 {
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

attributes #0 = { "profile-sample-accurate" "use-sample-profile" }
attributes #1 = { "use-sample-profile" }

; CHECK: ![[NOPROFILE_ID]] = !{!"function_entry_count", i64 -1}
; CHECK: ![[ZERO_ID]] = !{!"function_entry_count", i64 0}
; CHECK: ![[COLD_ID]] = !{!"function_section_prefix", !"unlikely"}
; UNKNOWN: ![[NOPROFILE_ID]] = !{!"function_entry_count", i64 -1}
; UNKNOWN: ![[UNKNOWN_ID]] = !{!"function_section_prefix", !"unknown"}
; ACCURATE: ![[ZERO_ID]] = !{!"function_entry_count", i64 0}
; ACCURATE: ![[COLD_ID]] = !{!"function_section_prefix", !"unlikely"}
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

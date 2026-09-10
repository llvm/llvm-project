; RUN: opt -passes='require<profile-summary>,inline' -S < %s | FileCheck %s
;
; When scaleProfData clamps a weight, the function that holds the scaled
; metadata is marked approxprofile.

declare void @side_effect()

define internal void @leaf_in_range() noinline {
entry:
  call void @side_effect()
  ret void
}

define internal void @mid_in_range() !prof !10 {
entry:
  call void @leaf_in_range(), !prof !11
  ret void
}

; CHECK-LABEL: define void @caller_in_range(){{[[:space:]]+}}!prof
; CHECK-NOT: #{{[0-9]+}}
; CHECK: call void @leaf_in_range(), !prof [[INRANGE:![0-9]+]]
define void @caller_in_range() !prof !12 {
entry:
  call void @mid_in_range(), !prof !13
  ret void
}

define internal void @leaf_overflow() noinline {
entry:
  call void @side_effect()
  ret void
}

define internal void @mid_overflow() !prof !20 {
entry:
  call void @leaf_overflow(), !prof !21
  ret void
}

; CHECK-LABEL: define void @caller_overflow()
; CHECK-SAME: #[[APPROX:[0-9]+]]
; CHECK: call void @leaf_overflow(), !prof [[CLAMPED:![0-9]+]]
define void @caller_overflow() !prof !22 {
entry:
  call void @mid_overflow(), !prof !23
  ret void
}

define internal void @approx_leaf() alwaysinline approxprofile {
entry:
  call void @side_effect()
  ret void
}

; CHECK-LABEL: define void @exact_caller()
; CHECK-SAME: #[[APPROX]]
define void @exact_caller() {
entry:
  call void @approx_leaf()
  ret void
}

; CHECK: attributes #[[APPROX]] = { approxprofile }
; CHECK: [[INRANGE]] = !{!"branch_weights", i32 5}
; CHECK: [[CLAMPED]] = !{!"branch_weights", i32 -1}

!llvm.module.flags = !{!30}
!30 = !{i32 1, !"ProfileSummary", !31}
!31 = !{!32, !33, !34, !35, !36, !37, !38, !39}
!32 = !{!"ProfileFormat", !"InstrProf"}
!33 = !{!"TotalCount", i64 400}
!34 = !{!"MaxCount", i64 100}
!35 = !{!"MaxInternalCount", i64 100}
!36 = !{!"MaxFunctionCount", i64 100}
!37 = !{!"NumCounts", i64 6}
!38 = !{!"NumFunctions", i64 6}
!39 = !{!"DetailedSummary", !40}
!40 = !{!41, !42, !43}
!41 = !{i32 10000, i64 100, i32 1}
!42 = !{i32 999000, i64 100, i32 2}
!43 = !{i32 999999, i64 100, i32 3}

!10 = !{!"function_entry_count", i64 100}
!11 = !{!"branch_weights", i32 5}
!12 = !{!"function_entry_count", i64 100}
!13 = !{!"branch_weights", i64 100}

!20 = !{!"function_entry_count", i64 100}
!21 = !{!"branch_weights", i64 8000000000}
!22 = !{!"function_entry_count", i64 100}
!23 = !{!"branch_weights", i64 100}

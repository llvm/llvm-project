; Test that branch weights and value profiles associated with invoke are updated
; in both caller and callee after inline, but invoke instructions with taken or
; not taken branch probabilities are not updated.
; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

declare i32 @__gxx_personality_v0(...)

define void @caller(ptr %func) personality ptr @__gxx_personality_v0 !prof !15 {
  call void @callee(ptr %func), !prof !16

  ret void
}

declare void @callee1(ptr %func)

declare void @callee2(ptr %func)

define void @callee(ptr %func) personality ptr @__gxx_personality_v0 !prof !17 {
  invoke void %func()
          to label %next unwind label %lpad, !prof !18

next:
  invoke void @callee1(ptr %func)
          to label %cont unwind label %lpad, !prof !19

cont:
  invoke void @callee2(ptr %func)
          to label %ret unwind label %lpad, !prof !20

lpad:
  %exn = landingpad {ptr, i32}
          cleanup
  unreachable

ret:
  ret void
}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 2000}
!8 = !{!"NumCounts", i64 2}
!9 = !{!"NumFunctions", i64 2}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 1000}
!16 = !{!"branch_weights", i64 1000}
!17 = !{!"function_entry_count", i32 1500}
!18 = !{!"VP", i32 0, i64 1500, i64 123, i64 900, i64 456, i64 600}
!19 = !{!"branch_weights", i32 1500}
!20 = !{!"branch_weights", i32 1234, i32 5678}

; CHECK-LABEL: @caller(
; CHECK:  invoke void %func(
; CHECK-NEXT: {{.*}} !prof ![[PROF1:[0-9]+]]
; CHECK:  invoke void @callee1(
; CHECK-NEXT: {{.*}} !prof ![[PROF2:[0-9]+]]
; CHECK:  invoke void @callee2(
; CHECK-NEXT: {{.*}} !prof ![[PROF3:[0-9]+]]

; CHECK-LABL: @callee(
; CHECK:  invoke void %func(
; CHECK-NEXT: {{.*}} !prof ![[PROF4:[0-9]+]]
; CHECK:  invoke void @callee1(
; CHECK-NEXT: {{.*}} !prof ![[PROF5:[0-9]+]]
; CHECK:  invoke void @callee2(
; CHECK-NEXT: {{.*}} !prof ![[PROF3]]


; CHECK: ![[PROF1]] = !{!"VP", i32 0, i64 1000, i64 123, i64 600, i64 456, i64 400}
; CHECK: ![[PROF2]] = !{!"branch_weights", i32 1000}
; CHECK: ![[PROF3]] = !{!"branch_weights", i32 1234, i32 5678}
; CHECK: ![[PROF4]] = !{!"VP", i32 0, i64 500, i64 123, i64 300, i64 456, i64 200}
; CHECK: ![[PROF5]] = !{!"branch_weights", i32 500}

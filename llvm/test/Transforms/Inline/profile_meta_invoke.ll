; Make sure that profile metadata is preserved when cloning a call.
; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

declare i32 @__gxx_personality_v0(...)

define void @callee(void ()* %func) !prof !15 {
  call void %func(), !prof !16
  ret void
}

define void @caller(void ()* %func) personality i32 (...)* @__gxx_personality_v0 {
  invoke void @callee(void ()* %func)
          to label %ret unwind label %lpad, !prof !17

ret:
  ret void

lpad:
  %exn = landingpad {i8*, i32}
          cleanup
  unreachable
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
!16 = !{!"VP", i32 0, i64 1000, i64 9191153033785521275, i64 400, i64 -1069303473483922844, i64 600}
!17 = !{!"branch_weights", i32 500}

; CHECK-LABEL: @caller(
; CHECK:  invoke void %func()
; CHECK-NEXT: {{.*}} !prof ![[PROF:[0-9]+]]
; CHECK: ![[PROF]] = !{!"VP", i32 0, i64 500, i64 9191153033785521275, i64 200, i64 -1069303473483922844, i64 300}

; Tests that instructions with value profiles and count-type branch weights are
; updated in both caller and callee after inline, but invoke instructions with
; taken or not taken branch probabilities are not updated.

; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

declare i32 @__gxx_personality_v0(...)

define void @caller(ptr %func) personality ptr @__gxx_personality_v0 !prof !15 {
  call void @callee(ptr %func), !prof !16

  ret void
}

declare void @callee1(ptr %func)

declare void @callee2(ptr %func)

define void @callee(ptr %obj) personality ptr @__gxx_personality_v0 !prof !17 {
  %vtable = load ptr, ptr %obj, !prof !21
  %func = load ptr, ptr %vtable
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
!21 = !{!"VP", i32 2, i64 1500, i64 789, i64 900, i64 321, i64 600}

; CHECK-LABEL: define void @caller(
; CHECK-SAME: ptr [[FUNC:%.*]]) personality ptr @__gxx_personality_v0 !prof [[PROF14:![0-9]+]] {
; CHECK-NEXT:    [[VTABLE_I:%.*]] = load ptr, ptr [[FUNC]], align 8, !prof [[PROF15:![0-9]+]]
; CHECK-NEXT:    [[FUNC_I:%.*]] = load ptr, ptr [[VTABLE_I]], align 8
; CHECK-NEXT:    invoke void [[FUNC_I]]()
; CHECK-NEXT:            to label %[[NEXT_I:.*]] unwind label %[[LPAD_I:.*]], !prof [[PROF16:![0-9]+]]
; CHECK:       [[NEXT_I]]:
; CHECK-NEXT:    invoke void @callee1(ptr [[FUNC_I]])
; CHECK-NEXT:            to label %[[CONT_I:.*]] unwind label %[[LPAD_I]], !prof [[PROF17:![0-9]+]]
; CHECK:       [[CONT_I]]:
; CHECK-NEXT:    invoke void @callee2(ptr [[FUNC_I]])
; CHECK-NEXT:            to label %[[CALLEE_EXIT:.*]] unwind label %[[LPAD_I]], !prof [[PROF18:![0-9]+]]
;

; CHECK-LABEL: define void @callee(
; CHECK-SAME: ptr [[OBJ:%.*]]) personality ptr @__gxx_personality_v0 !prof [[PROF19:![0-9]+]] {
; CHECK-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[OBJ]], align 8, !prof [[PROF20:![0-9]+]]
; CHECK-NEXT:    [[FUNC:%.*]] = load ptr, ptr [[VTABLE]], align 8
; CHECK-NEXT:    invoke void [[FUNC]]()
; CHECK-NEXT:            to label %[[NEXT:.*]] unwind label %[[LPAD:.*]], !prof [[PROF21:![0-9]+]]
; CHECK:       [[NEXT]]:
; CHECK-NEXT:    invoke void @callee1(ptr [[FUNC]])
; CHECK-NEXT:            to label %[[CONT:.*]] unwind label %[[LPAD]], !prof [[PROF22:![0-9]+]]
; CHECK:       [[CONT]]:
; CHECK-NEXT:    invoke void @callee2(ptr [[FUNC]])
; CHECK-NEXT:            to label %[[RET:.*]] unwind label %[[LPAD]], !prof [[PROF18]]

; CHECK: [[PROF14]] = !{!"function_entry_count", i64 1000}
; CHECK: [[PROF15]] = !{!"VP", i32 2, i64 1000, i64 789, i64 600, i64 321, i64 400}
; CHECK: [[PROF16]] = !{!"VP", i32 0, i64 1000, i64 123, i64 600, i64 456, i64 400}
; CHECK: [[PROF17]] = !{!"branch_weights", i32 1000}
; CHECK: [[PROF18]] = !{!"branch_weights", i32 1234, i32 5678}
; CHECK: [[PROF19]] = !{!"function_entry_count", i64 500}
; CHECK: [[PROF20]] = !{!"VP", i32 2, i64 500, i64 789, i64 300, i64 321, i64 200}
; CHECK: [[PROF21]] = !{!"VP", i32 0, i64 500, i64 123, i64 300, i64 456, i64 200}
; CHECK: [[PROF22]] = !{!"branch_weights", i32 500}

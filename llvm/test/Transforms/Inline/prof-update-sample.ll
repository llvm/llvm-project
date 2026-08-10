; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s
; Checks if inliner updates branch_weights annotation for call instructions.

declare void @ext();
declare void @ext1();
@func = global ptr null

; CHECK: define void @callee(i32 %n) !prof ![[ENTRY_COUNT:[0-9]*]]
define void  @callee(i32 %n) !prof !15 {
  %cond = icmp sle i32 %n, 10
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
; ext1 is optimized away, thus not updated.
; CHECK: call void @ext1(), !prof ![[COUNT_CALLEE1:[0-9]*]]
  call void @ext1(), !prof !16
  ret void
cond_false:
; ext is cloned and updated.
; CHECK: call void @ext(), !prof ![[COUNT_CALLEE:[0-9]*]]
  call void @ext(), !prof !16
  %f = load ptr, ptr @func
; CHECK: call void %f(), !prof ![[COUNT_IND_CALLEE:[0-9]*]] 
  call void %f(), !prof !18
  ret void
}

; CHECK: define void @caller()
define void @caller() {
; CHECK: call void @ext(), !prof ![[COUNT_CALLER:[0-9]*]]
; CHECK: call void %f.i(), !prof ![[COUNT_IND_CALLER:[0-9]*]]
  call void @callee(i32 15), !prof !17
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
!16 = !{!"branch_weights", i32 2000}
!17 = !{!"branch_weights", i32 400}
!18 = !{!"VP", i32 0, i64 140, i64 111, i64 80, i64 222, i64 40, i64 333, i64 20}
; CHECK: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 600}
; CHECK: ![[COUNT_CALLEE]] = !{!"branch_weights", i32 1200}
; CHECK: ![[COUNT_IND_CALLEE]] = !{!"VP", i32 0, i64 84, i64 111, i64 48, i64 222, i64 24, i64 333, i64 12}
; CHECK: ![[COUNT_CALLER]] = !{!"branch_weights", i32 800}
; CHECK: ![[COUNT_IND_CALLER]] = !{!"VP", i32 0, i64 56, i64 111, i64 32, i64 222, i64 16, i64 333, i64 8}

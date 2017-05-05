; RUN: opt < %s -inline -S | FileCheck %s
; Checks if inliner updates branch_weights annotation for call instructions.

declare void @ext();
declare void @ext1();
@func = global void ()* null

; CHECK: define void @callee(i32 %n) !prof ![[ENTRY_COUNT:[0-9]*]]
define void  @callee(i32 %n) !prof !1 {
  %cond = icmp sle i32 %n, 10
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
; ext1 is optimized away, thus not updated.
; CHECK: call void @ext1(), !prof ![[COUNT_CALLEE1:[0-9]*]]
  call void @ext1(), !prof !2
  ret void
cond_false:
; ext is cloned and updated.
; CHECK: call void @ext(), !prof ![[COUNT_CALLEE:[0-9]*]]
  call void @ext(), !prof !2
  %f = load void ()*, void ()** @func
; CHECK: call void %f(), !prof ![[COUNT_IND_CALLEE:[0-9]*]] 
  call void %f(), !prof !4
  ret void
}

; CHECK: define void @caller()
define void @caller() {
; CHECK: call void @ext(), !prof ![[COUNT_CALLER:[0-9]*]]
; CHECK: call void %f.i(), !prof ![[COUNT_IND_CALLER:[0-9]*]]
  call void @callee(i32 15), !prof !3
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MaxFunctionCount", i32 2000}
!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"branch_weights", i64 2000}
!3 = !{!"branch_weights", i64 400}
!4 = !{!"VP", i32 0, i64 140, i64 111, i64 80, i64 222, i64 40, i64 333, i64 20}
attributes #0 = { alwaysinline }
; CHECK: ![[ENTRY_COUNT]] = !{!"function_entry_count", i64 600}
; CHECK: ![[COUNT_CALLEE1]] = !{!"branch_weights", i64 2000}
; CHECK: ![[COUNT_CALLEE]] = !{!"branch_weights", i64 1200}
; CHECK: ![[COUNT_IND_CALLEE]] = !{!"VP", i32 0, i64 84, i64 111, i64 48, i64 222, i64 24, i64 333, i64 12}
; CHECK: ![[COUNT_CALLER]] = !{!"branch_weights", i64 800}
; CHECK: ![[COUNT_IND_CALLER]] = !{!"VP", i32 0, i64 56, i64 111, i64 32, i64 222, i64 16, i64 333, i64 8}

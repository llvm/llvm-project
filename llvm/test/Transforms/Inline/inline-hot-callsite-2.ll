; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -inline-threshold=0 -inlinehint-threshold=0 -hot-callsite-threshold=100 -S | FileCheck %s

; This tests that a callsite which is determined to be hot based on the caller's
; entry count and the callsite block frequency gets the hot-callsite-threshold.
; Another callsite with the same callee that is not hot does not get inlined
; because cost exceeds the inline-threshold. inlinthint-threshold is set to 0
; to ensure callee's hotness is not used to boost the threshold.

define i32 @callee1(i32 %x) !prof !21 {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  call void @extern()
  ret i32 %x3
}

define i32 @caller(i32 %n) !prof !22 {
; CHECK-LABEL: @caller(
  %cond = icmp sle i32 %n, 100
  br i1 %cond, label %cond_true, label %cond_false, !prof !0

cond_true:
; CHECK-LABEL: cond_true:
; CHECK-NOT: call i32 @callee1
; CHECK: ret i32 %x3.i
  %i = call i32 @callee1(i32 %n)
  ret i32 %i
cond_false:
; CHECK-LABEL: cond_false:
; CHECK: call i32 @callee1
; CHECK: ret i32 %j
  %j = call i32 @callee1(i32 %n)
  ret i32 %j
}
declare void @extern()

!0 = !{!"branch_weights", i32 64, i32 4}

!llvm.module.flags = !{!1}
!21 = !{!"function_entry_count", i64 200}
!22 = !{!"function_entry_count", i64 200}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
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

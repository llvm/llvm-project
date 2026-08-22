; RUN: opt < %s -passes="require<profile-summary>,tailcallelim" -disable-tail-call-elim-for-cold-calls=true -S | FileCheck %s --check-prefixes=CHECK,DISABLED
; RUN: opt < %s -passes="require<profile-summary>,tailcallelim" -disable-tail-call-elim-for-cold-calls=false -S | FileCheck %s --check-prefixes=CHECK,ENABLED

declare void @normal_callee()
declare void @cold_callee() cold

; Check that in SamplePGO, a cold function (with cold entry and cold blocks) has tail call elimination disabled when the flag is enabled.
define void @test_sample_cold_function() !prof !14 {
; CHECK-LABEL: @test_sample_cold_function(
; DISABLED: call void @normal_callee()
; ENABLED: tail call void @normal_callee()
  call void @normal_callee()
  ret void
}

; Check that in SamplePGO, a function with hot entry count IS marked as tail.
define void @test_sample_hot_function(i1 %cond) !prof !15 {
; CHECK-LABEL: @test_sample_hot_function(
; CHECK: tail call void @normal_callee()
entry:
  br i1 %cond, label %if.then, label %if.else, !prof !16

if.then:
  ret void

if.else:
  call void @normal_callee()
  ret void
}

; Check that in SamplePGO, a function with missing entry count but cold block weights has tail call elimination disabled when the flag is enabled.
define void @test_sample_missing_entry_cold_blocks(i1 %cond) {
; CHECK-LABEL: @test_sample_missing_entry_cold_blocks(
; DISABLED: call void @normal_callee()
; ENABLED: tail call void @normal_callee()
entry:
  br i1 %cond, label %if.then, label %if.else, !prof !19

if.then:
  ret void

if.else:
  call void @normal_callee()
  ret void
}

; Check that in SamplePGO, a function with missing entry count and missing block weights (unprofiled) IS marked as tail.
define void @test_sample_missing_counts() {
; CHECK-LABEL: @test_sample_missing_counts(
; CHECK: tail call void @normal_callee()
  call void @normal_callee()
  ret void
}

; Check that in SamplePGO, a function with missing entry count and hot block weights IS marked as tail.
define void @test_sample_missing_entry_hot_blocks(i1 %cond) {
; CHECK-LABEL: @test_sample_missing_entry_hot_blocks(
; CHECK: tail call void @normal_callee()
entry:
  br i1 %cond, label %if.then, label %if.else, !prof !16

if.then:
  ret void

if.else:
  call void @normal_callee()
  ret void
}

; Check that in SamplePGO, a function with partial profile (one block has cold weights, one block is unprofiled) IS marked as tail.
define void @test_sample_partial_profile(i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @test_sample_partial_profile(
; CHECK: tail call void @normal_callee()
entry:
  br i1 %cond1, label %bb1, label %bb2, !prof !19

bb1:
  br i1 %cond2, label %exit, label %unprofiled_block

unprofiled_block:
  call void @normal_callee()
  ret void

bb2:
  ret void

exit:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"SampleProfile"}
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
!14 = !{!"function_entry_count", i64 0}
!15 = !{!"function_entry_count", i64 1000}
!16 = !{!"branch_weights", i32 1000, i32 0}
!17 = !{!"function_entry_count", i64 1}
!18 = !{!"branch_weights", i32 1000, i32 1}
!19 = !{!"branch_weights", i32 1, i32 0}
